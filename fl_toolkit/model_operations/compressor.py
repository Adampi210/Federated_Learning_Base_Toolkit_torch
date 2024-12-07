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
    def __init__(self, split_strategy='top', split_ratio=1.0, alpha=1.0, beta=1.0):
        super().__init__(compression_ratio=split_ratio)
        self.split_strategy = split_strategy
        self.alpha = alpha
        self.beta = beta
        self.staleness = {}
        
    def _init_staleness(self, params):
        if len(self.staleness) == 0:
            for name, param in params.items():
                self.staleness[name] = torch.ones_like(param, dtype=torch.float)
    
    def _increment_staleness(self):
        for name in self.staleness:
            self.staleness[name] = self.staleness[name] + torch.ones_like(self.staleness[name])
    
    def _get_param_indices(self, param_names):
        n_params = len(param_names)
        n_split = int(n_params * self.compression_ratio)
        
        if self.split_strategy == 'top':
            return param_names[:n_split]
        elif self.split_strategy == 'bottom':
            return param_names[-n_split:]
        elif self.split_strategy == 'random_layers':
            return np.random.choice(param_names, n_split, replace=False)
        elif self.split_strategy in ['random_weights', 'stale_weights']:
            return param_names  # Handle all parameters with weight-level masks
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def compress(self, params):
        param_names = list(params.keys())
        selected_names = self._get_param_indices(param_names)
        
        compressed_params = {}
        masks = {}
        
        # Random per parameter mask
        if self.split_strategy == 'random_weights':
            for name, param in params.items():
                    mask = (torch.rand_like(param) < self.compression_ratio).float()
                    masks[name] = mask
                    compressed_params[name] = param * mask

        elif self.split_strategy == 'stale_weights':
            # Staleness handling
            self._init_staleness(params)
            self._increment_staleness()

            # Flatten all selected parameters and their staleness
            all_weights = []
            all_staleness = []
            shapes = {}
            indices_map = {}
            
            start_idx = 0
            for name in selected_names:
                w = params[name]
                s = self.staleness[name]
                numel = w.numel()
                shapes[name] = w.shape
                indices_map[name] = (start_idx, start_idx + numel)
                all_weights.append(w.view(-1))
                all_staleness.append(s.view(-1))
                start_idx += numel

            all_weights = torch.cat(all_weights)
            all_staleness = torch.cat(all_staleness)
            
            # Compute rewards
            rewards = self.alpha * all_staleness + self.beta * all_weights.abs()
            
            k = int(all_weights.numel() * self.compression_ratio)
            mask_vector = torch.zeros_like(all_weights)
            if k > 0:
                _, topk_indices = torch.topk(rewards, k, largest=True)
                mask_vector[topk_indices] = 1.0
            # Reshape masks back
            for name in selected_names:
                start_i, end_i = indices_map[name]
                param_mask = mask_vector[start_i:end_i].view(shapes[name])
                masks[name] = param_mask
                compressed_params[name] = params[name] * param_mask

            # Parameters not in selected_names get zero masks
            for name in param_names:
                if name not in selected_names:
                    masks[name] = torch.zeros_like(params[name])

            # Update staleness: zero out chosen parameters
            for name in masks:
                self.staleness[name] = self.staleness[name] * (1.0 - masks[name])
    
        # Layer wise compression, use selected layer names
        else:
            for name, param in params.items():
                if name in selected_names:
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

    def test_stale_weights():
        print("\nTesting stale_weights strategy")
        print("=" * 50)
        
        model = SimpleNet()
        # alpha=1.0, beta=1.0 as example
        splitter = SplitModelCompressor(split_strategy='stale_weights', split_ratio=0.5, alpha=1.0, beta=1.0)
        
        original_params = model.state_dict()
        print_model_params(original_params, "Original Parameters")

        # First compression
        compressed_params, masks = splitter.compress(original_params)
        print_model_params(compressed_params, "Compressed Parameters (Iteration 1)")
        print("\nMasks (Iteration 1):")
        for name, mask in masks.items():
            print(f"{name}:\n{mask}")

        # Let's call compress again to see staleness update
        # After first compression, staleness for chosen weights is reset to 0 and all are incremented
        compressed_params_2, masks_2 = splitter.compress(original_params)
        print_model_params(compressed_params_2, "Compressed Parameters (Iteration 2)")
        print("\nMasks (Iteration 2):")
        for name, mask in masks_2.items():
            print(f"{name}:\n{mask}")
            
          
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_pruning()
    # test_split_model()
    test_stale_weights()