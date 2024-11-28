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
    def compress(self, params, compression_rate, tolerance=0.01):
        compressed = {}
        metadata = {'U': {}, 'S': {}, 'V': {}, 'shapes': {}, 'orig_dims': {}}
        
        for name, param in params.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape
            
            if len(orig_shape) == 0:
                compressed[name] = param
                metadata['U'][name] = None
                metadata['S'][name] = None
                metadata['V'][name] = None
                metadata['orig_dims'][name] = None
                continue
            
            if len(orig_shape) == 1:
                param_2d = param.reshape(1, -1)
            else:
                param_2d = param.reshape(orig_shape[0], -1)
                
            metadata['orig_dims'][name] = param_2d.shape
            
            U, S, V = torch.svd(param_2d)
            
            min_dim = min(param_2d.shape)
            rank = max(1, int(min_dim * compression_rate))
            
            orig_size = param.numel() * param.element_size()
            left, right = 1, min_dim
            best_rank = rank
            best_ratio = float('inf')
            
            while left <= right:
                rank = (left + right) // 2
                compressed_size = (U[:, :rank].numel() + S[:rank].numel() + V[:, :rank].numel()) * param.element_size()
                current_ratio = compressed_size / orig_size
                
                if abs(current_ratio - compression_rate) < abs(best_ratio - compression_rate):
                    best_rank = rank
                    best_ratio = current_ratio
                    
                if current_ratio > compression_rate:
                    right = rank - 1
                else:
                    left = rank + 1
            
            rank = best_rank
            metadata['U'][name] = U[:, :rank]
            metadata['S'][name] = S[:rank]
            metadata['V'][name] = V[:, :rank]
            compressed[name] = U[:, :rank] * S[:rank]
            
        return compressed, metadata
    
    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, compressed in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            
            if len(orig_shape) == 0:
                decompressed[name] = compressed
                continue
            
            V = metadata['V'][name]
            orig_dim = metadata['orig_dims'][name]
            
            if target_scale is not None:
                new_shape = list(orig_shape)
                if len(orig_shape) >= 2:
                    new_shape[1] = int(orig_shape[1] * target_scale)
                else:
                    new_shape[0] = int(orig_shape[0] * target_scale)
                
                new_V_size = int(V.shape[0] * target_scale)
                new_V = torch.randn(new_V_size, V.shape[1], device=V.device)
                new_V = new_V / torch.norm(new_V, dim=0, keepdim=True)
                param = torch.mm(compressed, new_V.t())
            else:
                new_shape = orig_shape
                param = torch.mm(compressed, V.t())
            
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

class TensorTrainCompressor(BaseCompressor):
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def factorize_dim(self, dim):
        """Helper function to factorize dimension into roughly equal factors."""
        factors = []
        sqrt_dim = int(np.sqrt(dim))
        for i in range(sqrt_dim, 0, -1):
            if dim % i == 0:
                factors = [i, dim // i]
                break
        return factors if factors else [1, dim]

    def tt_decomposition(self, tensor, max_rank):
        """Decompose tensor into TT-format with adaptive ranks."""
        shape = tensor.shape
        if len(shape) == 1:
            return [tensor.reshape(1, -1, 1)]
        
        n, m = shape
        # Factorize the second dimension for better decomposition
        m1, m2 = self.factorize_dim(m)
        
        # Reshape and perform first decomposition
        tensor = tensor.reshape(n, m1, m2)
        n1 = n
        n2 = m1
        n3 = m2
        
        tensor = tensor.reshape(n1, -1)
        u1, s1, v1 = torch.svd(tensor)
        r1 = min(max_rank, len(s1))
        
        cores = []
        # First core
        cores.append(u1[:, :r1].reshape(-1, n1, r1))
        
        # Middle transformation
        temp = torch.mm(torch.diag(s1[:r1]), v1[:, :r1].t())
        temp = temp.reshape(r1, n2, n3)
        
        # Final core
        cores.append(temp.reshape(-1, n2, n3))
        
        return cores

    def tt_reconstruction(self, cores):
        """Reconstruct tensor from TT-cores."""
        result = cores[0]
        for core in cores[1:]:
            n1, r1, r2 = result.shape
            result = result.reshape(-1, r2)
            core_flat = core.reshape(r2, -1)
            result = torch.mm(result, core_flat)
        return result.reshape(cores[0].shape[1], -1)

    def compress(self, params, compression_rate, tolerance=0.01):
        compressed = {}
        metadata = {'shapes': {}, 'ranks': {}}
        
        for name, param in params.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape
            
            if len(orig_shape) == 0:
                compressed[name] = param
                continue
            
            # Reshape to 2D
            if len(orig_shape) == 1:
                param_2d = param.reshape(1, -1)
            else:
                param_2d = param.reshape(orig_shape[0], -1)
            
            orig_size = param.numel() * param.element_size()
            min_rank = 1
            max_rank = min(param_2d.shape)
            best_rank = max(1, int(np.sqrt(param_2d.numel() * compression_rate)))
            best_ratio = float('inf')
            
            # Binary search for the rank that gives desired compression ratio
            while min_rank <= max_rank:
                rank = (min_rank + max_rank) // 2
                
                # Try decomposition with current rank
                cores = self.tt_decomposition(param_2d, rank)
                
                # Calculate size with current rank
                compressed_size = sum(core.numel() * core.element_size() for core in cores)
                current_ratio = compressed_size / orig_size
                
                # Update best if closer to target
                if abs(current_ratio - compression_rate) < abs(best_ratio - compression_rate):
                    best_rank = rank
                    best_ratio = current_ratio
                    best_cores = cores
                
                if current_ratio > compression_rate:
                    max_rank = rank - 1
                else:
                    min_rank = rank + 1
            
            # Use best found rank
            compressed[name] = best_cores
            metadata['ranks'][name] = best_rank
        
        return compressed, metadata

    def get_scaling_matrix(self, in_dim, out_dim, device):
        return torch.randn(in_dim, out_dim, device=device) / np.sqrt(out_dim)

    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, cores in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            
            if len(orig_shape) == 0:
                decompressed[name] = cores
                continue
            
            # Reconstruct
            param = self.tt_reconstruction(cores)
            
            if target_scale is not None:
                if len(orig_shape) == 1:
                    new_shape = [int(orig_shape[0] * target_scale)]
                    scaling_matrix = self.get_scaling_matrix(param.shape[1], new_shape[0], param.device)
                else:
                    new_shape = list(orig_shape)
                    new_shape[1] = int(orig_shape[1] * target_scale)
                    scaling_matrix = self.get_scaling_matrix(param.shape[1], new_shape[1], param.device)
                
                param = torch.mm(param, scaling_matrix)
            
            param = param.reshape(new_shape)
            decompressed[name] = param
            
        return decompressed

    def validate_compression(self, original_params, compressed_params, metadata,
                       reconstructed_params, compression_rate, tolerance=1e-5):
        validation = super().validate_compression(
            original_params, compressed_params, reconstructed_params,
            compression_rate, tolerance)
        
        # Calculate original size
        orig_size = sum(p.numel() * p.element_size() for p in original_params.values())
        
        # Calculate compressed size, handling both tensor cores and scalar values
        comp_size = 0
        for param in compressed_params.values():
            if isinstance(param, list):  # List of tensor cores
                comp_size += sum(core.numel() * core.element_size() for core in param)
            else:  # Scalar or uncompressed parameter
                comp_size += param.numel() * param.element_size()
        
        # Calculate and store compression metrics
        actual_rate = comp_size / orig_size
        validation['compression_ratio'] = actual_rate
        validation['target_ratio_met'] = abs(actual_rate - compression_rate) < tolerance
        
        # Store tensor train specific metrics
        validation['tt_ranks'] = {
            name: metadata['ranks'][name] 
            for name in original_params.keys() 
            if len(original_params[name].shape) > 0
        }
        
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
        ("Tensor Train", TensorTrainCompressor())
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
                comp_size = 0
                for param in compressed_params.values():
                    if isinstance(param, list):
                        comp_size += sum(core.numel() * core.element_size() for core in param)
                    else:
                        comp_size += param.numel() * param.element_size()

                print(f"Original size: {orig_size/1024:.2f} KB")
                print(f"Compressed size: {comp_size/1024:.2f} KB")
                print(f"Total compressed size: {comp_size/1024:.2f} KB")
                print(f"Actual compression ratio: {comp_size/orig_size:.4f}")

                # Print compression-specific metrics
                if isinstance(compressor, SVDLiteCompressor):
                    avg_sv_kept = np.mean(list(validation['singular_values_kept'].values()))
                    print(f"Average singular values kept: {avg_sv_kept:.4f}")
                elif isinstance(compressor, TensorTrainCompressor):
                    avg_rank = np.mean(list(validation['tt_ranks'].values()))
                    print(f"Average TT rank: {avg_rank:.4f}")