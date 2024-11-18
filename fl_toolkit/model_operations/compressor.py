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

        orig_size = sum(p.numel() * p.element_size() for p in original_params.values())
        comp_size = sum(p.numel() * p.element_size() for p in compressed_params.values())
        actual_rate = comp_size / orig_size
        validation['compression_ratio'] = actual_rate
        validation['target_ratio'] = compression_rate
        validation['target_ratio_met'] = abs(actual_rate - compression_rate) < tolerance
        
        total_error = 0
        for key in original_params:
            error = torch.norm(original_params[key] - reconstructed_params[key]) / torch.norm(original_params[key])
            total_error += error.item()
        validation['reconstruction_error'] = total_error / len(original_params)
       
        return validation
    
# Projects using a random projection matrix
# Works by the Johnson-Lindenstrauss lemma
class RandomProjectionCompressor(BaseCompressor):
    def compress(self, params, compression_rate):
        compressed = {}
        metadata = {'projections': {}, 'shapes': {}}
        
        for name, param in params.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape
            
            # Reshape to 2D if needed
            if len(orig_shape) > 2:
                param_2d = param.reshape(orig_shape[0], -1)
            else:
                param_2d = param
            
            # Create random projection matrix
            in_dim = param_2d.shape[1]
            out_dim = max(1, int(in_dim * compression_rate))
            projection = torch.randn(in_dim, out_dim, device=param.device) / np.sqrt(out_dim)

            metadata['projections'][name] = projection
            
            compressed[name] = torch.mm(param_2d, projection)
            
        return compressed, metadata

    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, compressed in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            projection = metadata['projections'][name]
            
            if target_scale is not None:
                rec_dim = int(projection.shape[0] * target_scale)
                rec_projection = torch.randn(projection.shape[1], rec_dim, 
                                             device=compressed.device) / np.sqrt(rec_dim)
            else:
                rec_projection = projection
                
            # Reconstruct
            param = torch.mm(compressed, rec_projection.t())
            
            # Reshape if needed
            if len(orig_shape) > 2:
                param = param.reshape(orig_shape[0], -1, *orig_shape[2:])
            
            decompressed[name] = param
        return decompressed
    
    def validate_compression(self, original_params, compressed_params, 
                             reconstructed_params, compression_rate, tolerance=1e-5):
        return super().validate_compression(
            original_params, compressed_params, reconstructed_params,
            compression_rate, tolerance)
        
# Uses SVD to compress parameter size
class SVDLiteCompressor(BaseCompressor):
    def compress(self, params, compression_rate):
        compressed = {}
        metadata = {'U': {}, 'S': {}, 'V': {}, 'shapes': {}}
        
        for name, param in param.items():
            orig_shape = param.shape
            metadata['shapes'][name] = orig_shape

            # Reshape to 2D if needed
            if len(orig_shape) > 2:
                    param_2d = param.reshape(orig_shape[0], -1)
            else:
                param_2d = param
                
            # SVD
            U, S, V = torch.svd(param_2d)
            
            rank = max(1, int(min(param_2d.shape) * compression_rate))
            
            metadata['U'][name] = U[:, :rank]
            metadata['S'][name] = S[:rank]
            metadata['V'][name] = V[:, :rank]

            compressed[name] = torch.mm(U[:, :rank] * S[:rank], V[:, :rank].t())
            
            if len(orig_shape) > 2:
                compressed[name] = compressed[name].reshape(orig_shape)
        
        return compressed, metadata
    
    def decompress(self, compressed_params, metadata, target_scale=None):
        decompressed = {}
        
        for name, _ in compressed_params.items():
            orig_shape = metadata['shapes'][name]
            U = metadata['U'][name]
            S = metadata['S'][name]
            V = metadata['V'][name]
            
            if target_scale is not None:
                S = S * target_scale
                
            param = torch.mm(U * S, V.t())
            
            if len(orig_shape) > 2:
                param = param.reshape(orig_shape)
                
            decompressed[name] = param
            
        return decompressed
    
    def validate_compression(self, original_params, compressed_params, 
                             reconstructed_params, compression_rate, tolerance=1e-5):
        validation = super().validate_compression(
            original_params, compressed_params, reconstructed_params,
            compression_rate, tolerance)
        validation['singular_values_kept'] = {}
        
        for name in original_params:
            _, S, _ = torch.svd(original_params[name].reshape(original_params[name].shape[0], -1))
            validation['singular_values_kept'][name] = len(self.metadata['S'][name]) / len(S)
        return validation
        
class DCTCompressor(BaseCompressor):
    pass