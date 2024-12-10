# data_operations/drift_handlers.py

import random
from torch.utils.data import Subset
from torchvision import transforms
from enum import Enum

# PACS Drift
class DriftStrategy(Enum):
    ADD = "add"  # Add new domain samples
    REPLACE = "replace"  # Replace existing samples with new domain
    MIXED = "mixed"  # Mix of add and replace based on size constraints

class PACSDomainDrift:
    def __init__(self, source_domains, target_domains, drift_rate, desired_size=None):
        self.source_domains = source_domains
        self.target_domains = target_domains
        self.drift_rate = drift_rate
        self.desired_size = desired_size
        self.used_source_indices = set()
        self.used_target_indices = set()
    
    def _get_domain_indices(self, dataset, domains):
        indices = []
        for domain in domains:
            domain_indices = [i for i in range(len(dataset)) if dataset[i][2] == domain]
            indices.extend(domain_indices)
        return indices
    
    def _sample_indices(self, available_indices, n):
        if n <= 0:
            return []
        n = min(n, len(available_indices))
        return random.sample(list(available_indices), n)
    
    def apply(self, dataset):
        # First call - initialize with source domains
        if not self.used_source_indices and not self.used_target_indices:
            source_indices = self._get_domain_indices(dataset, self.source_domains)
            if len(source_indices) == 0:
                raise ValueError("No source domain samples available to initialize drift.")
            initial_size = len(source_indices) if self.desired_size is None else min(len(source_indices), self.desired_size)
            self.used_source_indices = set(random.sample(source_indices, initial_size))
            return Subset(dataset, list(self.used_source_indices))
        
        current_size = len(self.used_source_indices) + len(self.used_target_indices)
        drift_samples = int(current_size * self.drift_rate)
        
        # Get available indices from target domains
        all_target_indices = set(self._get_domain_indices(dataset, self.target_domains))
        available_target = all_target_indices - self.used_target_indices
        
        # If no available target samples left, reuse existing target indices
        if not available_target:
            available_target = all_target_indices
            print("No available target samples left. Reusing existing target indices.")
        
        if self.desired_size is None or current_size < self.desired_size:
            # Case 1: Can add more samples
            space_left = float('inf') if self.desired_size is None else self.desired_size - current_size
            
            if drift_samples + current_size <= (float('inf') if self.desired_size is None else self.desired_size):
                # Add all drift samples
                new_samples = self._sample_indices(available_target, drift_samples)
                self.used_target_indices.update(new_samples)
                all_indices = list(self.used_source_indices | self.used_target_indices)
            else:
                # Add what we can, replace the rest
                add_samples = self._sample_indices(available_target, int(space_left))
                self.used_target_indices.update(add_samples)
                
                to_replace = drift_samples - len(add_samples)
                
                if self.used_source_indices:
                    # Replace from source indices if available
                    replace_indices = self._sample_indices(self.used_source_indices, to_replace)
                    self.used_source_indices -= set(replace_indices)
                else:
                    # Replace from target indices if no source indices left
                    replace_indices = self._sample_indices(self.used_target_indices, to_replace)
                    self.used_target_indices -= set(replace_indices)
                
                new_replace = self._sample_indices(available_target, len(replace_indices))
                self.used_target_indices.update(new_replace)
                all_indices = list(self.used_source_indices | self.used_target_indices)
        else:
            # Case 2: At desired size, just replace
            if self.used_source_indices:
                # Replace from source indices if available
                replace_indices = self._sample_indices(self.used_source_indices, drift_samples)
                self.used_source_indices -= set(replace_indices)
            else:
                # Replace from target indices if no source indices left
                replace_indices = self._sample_indices(self.used_target_indices, drift_samples)
                self.used_target_indices -= set(replace_indices)
            
            new_samples = self._sample_indices(available_target, len(replace_indices))
            self.used_target_indices.update(new_samples)
            all_indices = list(self.used_source_indices | self.used_target_indices)
        
        # Ensure all_indices are within the dataset range
        valid_indices = [idx for idx in all_indices if 0 <= idx < len(dataset)]
        if len(valid_indices) != len(all_indices):
            print(f"Warning: Some indices were out of range and have been removed. Valid indices count: {len(valid_indices)}")
        
        return Subset(dataset, valid_indices)
