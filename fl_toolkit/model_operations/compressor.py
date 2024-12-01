# model_operations/compression.py

import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseCompressor(ABC):
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
    
    @abstractmethod
    def compress(self, params):
        pass
        
    @abstractmethod
    def decompress(self, compressed_params):
        pass

# Magnitude-based weight pruning compression
class WeightPruner(BaseCompressor):
    def __init__(self, compression_ratio=0.5):
        super().__init__(compression_ratio)
        self.masks = {}

    def compress(self, params):
        compressed_params = {}
        masks = {}
        
        for name, param in params.items():
            tensor = param.data
            k = int(tensor.numel() * (1 - self.compression_ratio))
            if k == 0:
                threshold = -1
            else:
                threshold = torch.kthvalue(tensor.abs().view(-1), k)[0]
            
            mask = (tensor.abs() > threshold).float()
            masks[name] = mask
            compressed_params[name] = tensor * mask
            
                
        return compressed_params, masks
    
    def recompress(self, updated_params, masks=None):
        if masks is None:
            return updated_params
        
        recompressed = {}
        for name, param in updated_params.items():
            if name in masks:
                recompressed[name] = param * masks[name]
            else:
                recompressed[name] = param
        return recompressed    
    
    def decompress(self, compressed_params, masks=None):
        # Cannot inverse pruning
        return compressed_params

# Selective model splitting
class SplitModelCompressor(BaseCompressor):
    def __init__(self, split_strategy='top', split_ratio=1.0):
        super().__init__(compression_ratio=split_ratio)
        self.split_strategy = split_strategy
        
    def _get_param_indices(self, param_names):
        n_params = len(param_names)
        n_split = int(n_params * self.compression_ratio)
        
        if self.split_strategy == 'top':
            return param_names[:n_split]
        elif self.split_strategy == 'bottom':
            return param_names[-n_split:]
        elif self.split_strategy == 'random_layers':
            return np.random.choice(param_names, n_split, replace=False)
        elif self.split_strategy == 'random_weights':
            return param_names  # Handle all parameters with weight-level masks
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def compress(self, params):
        param_names = list(params.keys())
        selected_names = self._get_param_indices(param_names)
        
        compressed_params = {}
        masks = {}
        
        for name, param in params.items():
            if self.split_strategy == 'random_weights':
                # Create random mask for each parameter
                mask = torch.rand_like(param) < self.compression_ratio
                
                masks[name] = mask.float()
                compressed_params[name] = param * mask.float()
            elif name in selected_names:
                compressed_params[name] = param.clone()
                masks[name] = torch.ones_like(param).float()
            else:
                masks[name] = torch.zeros_like(param).float()
        
        return compressed_params, masks
        
    def recompress(self, updated_params, masks=None):
        if masks is None:
            return updated_params
        
        recompressed = {}
        for name, param in updated_params.items():
            if name in masks:
                recompressed[name] = param * masks[name]
            else:
                recompressed[name] = param
        return recompressed
    
    def decompress(self, compressed_params):
        # Cannot decompress after split
        return compressed_params
     
if __name__ == '__main__':
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 2)
            self.layer2 = nn.Linear(2, 2)
            
            # Initialize with known values for easier testing
            with torch.no_grad():
                self.layer1.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                self.layer1.bias.data = torch.tensor([0.1, 0.2])
                self.layer2.weight.data = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
                self.layer2.bias.data = torch.tensor([0.3, 0.4])
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    def print_model_params(params, title="Model Parameters"):
        print(f"\n{title}")
        print("-" * 50)
        for name, param in params.items():
            print(f"{name}:\n{param}")

    def test_pruning():
        print("\nTesting Weight Pruner")
        print("=" * 50)
        
        model = SimpleNet()
        pruner = WeightPruner(compression_ratio=1.0)  # Keep 50% of weights
        
        original_params = model.state_dict()
        print_model_params(original_params, "Original Parameters")
        
        compressed_params, masks = pruner.compress(original_params)
        print_model_params(compressed_params, "Pruned Parameters")
        
        # Print masks
        print("\nPruning Masks:")
        for name, mask in masks.items():
            print(f"{name}:\n{mask}")

    def test_split_model():
        print("\nTesting Split Model Compressor")
        print("=" * 50)
        
        model = SimpleNet()
        strategies = ['top', 'bottom', 'random_layers', 'random_weights']
        
        for strategy in strategies:
            print(f"\nTesting {strategy} split strategy")
            print("-" * 30)
            
            splitter = SplitModelCompressor(split_strategy=strategy, split_ratio=0.25)
            
            original_params = model.state_dict()
            compressed_params, masks = splitter.compress(original_params)
            
            print(f"Original parameter names: {list(original_params.keys())}")
            print(f"Selected parameter names: {list(compressed_params.keys())}")
            print_model_params(compressed_params, f"Split Parameters ({strategy})")
            print("\nSplit Masks:")
            for name, mask in masks.items():
                print(f"{name}:\n{mask}")

            
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_pruning()
    test_split_model()