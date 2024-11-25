# model_operations/base_compressor.py

import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, params, compression_rate):
        pass
    
    @abstractmethod
    def decompress(self, compressed_params, metadata, target_scale):
        pass
    
    @abstractmethod
    def validate_compression(self, original_params,
                             compressed_params,
                             reconstructed_params, 
                             compression_rate,
                             tolerance=1e-5):
        validation = {}
        return validation
    
# Projects using a random projection matrix
# Works by the Johnson-Lindenstrauss lemma
class RandomProjectionCompressor(BaseCompressor):
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)  # Set seed for reproducibility
        
    def get_projection_matrix(self, in_dim, out_dim, device):
        # Given same input dims and seed, will generate same projection matrix
        return torch.randn(in_dim, out_dim, device=device) / np.sqrt(out_dim)
    
    def compress(self, params, compression_rate):
        compressed = {}
        metadata = {'shapes': {}, 'proj_dims': {}}  # Store only dimensions, not matrices
        
        for name, param in params.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape
            # Pass through scalar values
            if len(orig_shape) == 0:
                compressed[name] = param
                metadata['proj_dims'][name] = None
                continue
            
            # Reshape to 2D
            if len(orig_shape) == 1:
                param_2d = param.reshape(1, -1)
            else:
                param_2d = param.reshape(orig_shape[0], -1)
            
            # Create and store projection dimensions
            in_dim = param_2d.shape[1]
            out_dim = max(1, int(in_dim * compression_rate))
            metadata['proj_dims'][name] = (in_dim, out_dim)
            
            # Get projection matrix
            projection = self.get_projection_matrix(in_dim, out_dim, param.device)
            
            compressed[name] = torch.mm(param_2d, projection)
        return compressed, metadata

    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, compressed in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            # Pass through scalar values
            if len(orig_shape) == 0:
                decompressed[name] = compressed
                continue
            
            proj_dims = metadata['proj_dims'][name]
            in_dim, out_dim = proj_dims
            
            if target_scale is not None:
                if abs(target_scale - out_dim/in_dim) < 1e-6:
                    decompressed[name] = compressed
                    continue
                
                # Scale the original shape
                new_shape = list(orig_shape)
                if len(orig_shape) >= 2:
                    new_shape[1] = int(orig_shape[1] * target_scale)
                else:
                    new_shape[0] = int(orig_shape[0] * target_scale)
                
                # Get reconstruction projection
                new_dim = int(in_dim * target_scale)
                rec_projection = self.get_projection_matrix(out_dim, new_dim, compressed.device)
            else:
                new_shape = orig_shape
                # Get original projection
                projection = self.get_projection_matrix(in_dim, out_dim, compressed.device)
                rec_projection = projection.t()
            
            # Reconstruct
            param = torch.mm(compressed, rec_projection)
            param = param.reshape(new_shape)
            decompressed[name] = param
            
        return decompressed
    
    def validate_compression(self, original_params, compressed_params, metadata, 
                           reconstructed_params, compression_rate, tolerance=1e-5):
        validation = super().validate_compression(
            original_params, compressed_params, reconstructed_params,
            compression_rate, tolerance)
        
        # Only count compressed parameters size, not projections
        orig_size = sum(p.numel() * p.element_size() for p in original_params.values())
        comp_size = sum(p.numel() * p.element_size() for p in compressed_params.values())

        actual_rate = comp_size / orig_size
        validation['compression_ratio'] = actual_rate
        validation['target_ratio_met'] = abs(actual_rate - compression_rate) < tolerance
        
        return validation      

# Uses SVD to compress parameter size
class SVDLiteCompressor(BaseCompressor):
    def compress(self, params, compression_rate):
        compressed = {}
        metadata = {'U': {}, 'S': {}, 'V': {}, 'shapes': {}, 'orig_dims': {}}  # Added orig_dims
        
        for name, param in params.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape
            
            # Pass through scalar values
            if len(orig_shape) == 0:
                compressed[name] = param
                metadata['U'][name] = None
                metadata['S'][name] = None
                metadata['V'][name] = None
                metadata['orig_dims'][name] = None
                continue
            
            # Reshape to 2D
            if len(orig_shape) == 1:
                param_2d = param.reshape(1, -1)
            else:
                param_2d = param.reshape(orig_shape[0], -1)
                
            # Store original dimensions
            metadata['orig_dims'][name] = param_2d.shape[1]
                
            # SVD
            U, S, V = torch.svd(param_2d)
            rank = max(1, int(min(param_2d.shape) * compression_rate))
            
            # Store truncated components
            metadata['U'][name] = U[:, :rank]
            metadata['S'][name] = S[:rank]
            metadata['V'][name] = V[:, :rank]
            
            # Store compressed form (just low rank factors)
            compressed[name] = U[:, :rank] * S[:rank]  # Store U*S as compressed
            
        return compressed, metadata
    
    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, compressed in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            
            # Pass through scalar values
            if len(orig_shape) == 0:
                decompressed[name] = compressed
                continue
            
            # Get components
            V = metadata['V'][name]
            orig_dim = metadata['orig_dims'][name]
            
            if target_scale is not None:
                # Calculate new dimensions
                new_shape = list(orig_shape)
                if len(orig_shape) >= 2:
                    new_shape[1] = int(orig_shape[1] * target_scale)
                else:
                    new_shape[0] = int(orig_shape[0] * target_scale)
                
                # Scale the original dimension
                new_dim = int(orig_dim * target_scale)
                
                # Create new V matrix with scaled size
                new_V = torch.randn(new_dim, V.shape[1], device=V.device)
                new_V = new_V / torch.norm(new_V, dim=0, keepdim=True)
                
                # Reconstruct with scaled V
                param = torch.mm(compressed, new_V.t())
            else:
                # Normal reconstruction
                param = torch.mm(compressed, V.t())
            
            # Reshape to target shape
            param = param.reshape(new_shape)
            decompressed[name] = param
            
        return decompressed

    def validate_compression(self, original_params, compressed_params, metadata, 
                           reconstructed_params, compression_rate, tolerance=1e-5):
        validation = super().validate_compression(
            original_params, compressed_params, reconstructed_params,
            compression_rate, tolerance)
        
        # Add SVD-specific metrics
        validation['singular_values_kept'] = {}
        for name in original_params:
            if len(original_params[name].shape) > 0:
                _, S, _ = torch.svd(original_params[name].reshape(original_params[name].shape[0], -1))
                validation['singular_values_kept'][name] = len(metadata['S'][name]) / len(S)
        
        # Calculate compression ratio using compressed params and V matrices
        orig_size = sum(p.numel() * p.element_size() for p in original_params.values())
        comp_size = sum(p.numel() * p.element_size() for p in compressed_params.values())
        meta_size = sum(v.numel() * v.element_size() if v is not None else 0 
                       for v in metadata['V'].values())
        
        actual_rate = (comp_size + meta_size) / orig_size
        validation['compression_ratio'] = actual_rate
        validation['target_ratio_met'] = abs(actual_rate - compression_rate) < tolerance
        
        return validation
    
class DCTCompressor(BaseCompressor):
   def dct2d(self, x):
       X1 = torch.fft.rfft(x, dim=0, norm='ortho')
       X2 = torch.fft.rfft(X1.real, dim=1, norm='ortho')
       return X2
       
   def idct2d(self, X):
       x1 = torch.fft.irfft(X, dim=1, norm='ortho')
       x2 = torch.fft.irfft(x1, dim=0, norm='ortho')
       return x2
   
   def compress(self, params, compression_rate):
       compressed = {}
       metadata = {'shapes': {}, 'frequency_shapes': {}}
       
       for name, param in params.items():
           orig_shape = param.shape
           metadata['shapes'][name] = orig_shape
           
           # Pass through scalar values
           if len(orig_shape) == 0:
               compressed[name] = param
               metadata['frequency_shapes'][name] = None
               continue
           
           # Reshape to 2D
           if len(orig_shape) == 1:
               param_2d = param.reshape(1, -1)
           else:
               param_2d = param.reshape(orig_shape[0], -1)
               
           # Apply DCT
           freq_coeffs = self.dct2d(param_2d)
           
           # Keep top k coefficients based on magnitude
           k = max(1, int(freq_coeffs.numel() * compression_rate))
           values, indices = torch.topk(freq_coeffs.abs().flatten(), k)
           
           # Create sparse tensor of kept coefficients
           compressed_coeffs = torch.zeros_like(freq_coeffs.flatten())
           compressed_coeffs[indices] = freq_coeffs.flatten()[indices]
           compressed_coeffs = compressed_coeffs.reshape(freq_coeffs.shape)
           
           metadata['frequency_shapes'][name] = compressed_coeffs.shape
           compressed[name] = compressed_coeffs
           
       return compressed, metadata
   
   def decompress(self, compressed_params, metadata, target_scale=None):
       decompressed = {}
       
       for name, compressed in compressed_params.items():
           orig_shape = metadata['shapes'][name]
           
           # Pass through scalar values
           if len(orig_shape) == 0:
               decompressed[name] = compressed
               continue
           
           if target_scale is not None:
               # Scale the frequency coefficients
               compressed = compressed * target_scale
               
               # Scale the original shape
               new_shape = list(orig_shape)
               if len(orig_shape) >= 2:
                   new_shape[1] = int(orig_shape[1] * target_scale)
               else:
                   new_shape[0] = int(orig_shape[0] * target_scale)
           else:
               new_shape = orig_shape
           
           # Inverse DCT
           param = self.idct2d(compressed)
           param = param.reshape(new_shape)
           decompressed[name] = param
           
       return decompressed

   def validate_compression(self, original_params, compressed_params, metadata,
                          reconstructed_params, compression_rate, tolerance=1e-5):
       validation = super().validate_compression(
           original_params, compressed_params, reconstructed_params,
           compression_rate, tolerance)
       
       # Add frequency-domain metrics
       validation['frequency_energy_retention'] = {}
       for name in original_params:
           if len(original_params[name].shape) > 0:  # Skip scalar values
               param_2d = original_params[name].reshape(original_params[name].shape[0], -1)
               orig_freq = self.dct2d(param_2d)
               comp_freq = compressed_params[name]
               validation['frequency_energy_retention'][name] = (
                   torch.norm(comp_freq) / torch.norm(orig_freq)
               ).item()
       
       # Calculate compression ratio
       orig_size = sum(p.numel() * p.element_size() for p in original_params.values())
       comp_size = sum(p.numel() * p.element_size() for p in compressed_params.values())
       
       actual_rate = comp_size / orig_size
       validation['compression_ratio'] = actual_rate
       validation['target_ratio_met'] = abs(actual_rate - compression_rate) < tolerance
       
       return validation

if __name__ == "__main__":
    from fl_toolkit import *
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    class SmallMLP(BaseModelArchitecture):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            
        def forward(self, x):
            x = self.flatten(x)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return nn.functional.log_softmax(x, dim=1)

    class MediumMLP(BaseModelArchitecture):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 64)
            self.fc5 = nn.Linear(64, 10)
            
        def forward(self, x):
            x = self.flatten(x)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            x = nn.functional.relu(self.fc4(x))
            x = self.fc5(x)
            return nn.functional.log_softmax(x, dim=1)

    class LargeMLP(BaseModelArchitecture):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, 256)
            self.fc2 = nn.Linear(256, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, 256)
            self.fc5 = nn.Linear(256, 10)
            
        def forward(self, x):
            x = self.flatten(x)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            x = nn.functional.relu(self.fc4(x))
            x = self.fc5(x)
            return nn.functional.log_softmax(x, dim=1)


        
    # Model configurations
    model_configs = [
        ("Small MLP", SmallMLP()),
        ("Medium MLP", MediumMLP()), 
        ("Large MLP", LargeMLP()), 
    ]
    
    # Compression configurations
    compression_configs = [
        ("Random Projection", RandomProjectionCompressor()),
        ("SVD Lite", SVDLiteCompressor()),
        # ("DCT", DCTCompressor())
    ]
    
    # Test parameters
    compression_rates = [0.1, 0.3, 0.5, 0.9]
    expansion_scale = 1.5
    
    for model_name, model in model_configs:
        print(f"\n{'='*20} Testing {model_name} {'='*20}")
        
        # Get model parameters
        params = model.state_dict()
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in params.values())
        print(f"Total parameters: {total_params:,}")
        
        for comp_name, compressor in compression_configs:
            print(f"\n{'-'*10} {comp_name} Compression {'-'*10}")
            
            for rate in compression_rates:
                print(f"\nCompression rate: {rate}")
                
                # Compress
                compressed_params, metadata = compressor.compress(params, rate)
                # Test normal reconstruction
                reconstructed_params = compressor.decompress(compressed_params, metadata, target_scale=1)
                validation = compressor.validate_compression(
                    params, compressed_params, metadata, reconstructed_params, rate
                )
                
                # Calculate sizes
                orig_size = sum(p.numel() * p.element_size() for p in params.values())
                comp_size = sum(p.numel() * p.element_size() for p in compressed_params.values())

                print(f"Original size: {orig_size/1024:.2f} KB")
                print(f"Compressed size: {comp_size/1024:.2f} KB")
                print(f"Total compressed size: {(comp_size)/1024:.2f} KB")
                print(f"Actual compression ratio: {(comp_size)/orig_size:.4f}")
                
            # Test expansion
            expanded_params = compressor.decompress(
                compressed_params, metadata, target_scale=expansion_scale
            )
            
            # Print compression-specific metrics
            if isinstance(compressor, SVDLiteCompressor):
                avg_sv_kept = np.mean(list(validation['singular_values_kept'].values()))
                print(f"Average singular values kept: {avg_sv_kept:.4f}")
            elif isinstance(compressor, DCTCompressor):
                avg_energy = np.mean(list(validation['frequency_energy_retention'].values()))
                print(f"Average frequency energy retained: {avg_energy:.4f}")
            