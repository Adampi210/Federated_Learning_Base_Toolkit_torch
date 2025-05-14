# data_operations/drift_handlers.py

import random
import torch
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import PIL
from PIL import Image
from enum import Enum

# Drift Class
class DomainDrift:
    def __init__(self, data_handler, source_domains, target_domains, drift_rate, desired_size=None):
        self.dataset = data_handler.dataset
        self.source_domains = source_domains
        self.target_domains = target_domains
        self.drift_rate = drift_rate
        self.desired_size = desired_size
        self.domain_array = np.array([domain for _, _, domain in self.dataset])
        self.domain_to_indices = {d: np.where(self.domain_array == d)[0] for d in set(self.domain_array)}
        self.set_target_domains(target_domains)
        self.current_indices = None
    
    def set_target_domains(self, new_target_domains):
        self.target_domains = new_target_domains
        self.target_indices = np.concatenate([self.domain_to_indices[d] for d in self.target_domains if d in self.domain_to_indices])
    
    def apply(self):
        # Create initial subset from source domains
        if self.current_indices is None:
            print('Initial dataset creation')
            source_indices =  np.concatenate([self.domain_to_indices[d] for d in self.source_domains if d in self.domain_to_indices])
            if len(source_indices) == 0:
                raise ValueError("No source indices found for the specified source domains.")
            if self.desired_size is None:
                initial_size = int(len(source_indices))
            else:
                initial_size = min(self.desired_size, len(source_indices))
            self.current_indices = np.random.choice(source_indices, initial_size, replace=False)
            return Subset(self.dataset, self.current_indices.tolist())
        else:
            # Skip if drift rate is 0
            if self.drift_rate <= 0:
                return Subset(self.dataset, self.current_indices.tolist())
            # Then identify non-target samples in the current subset
            current_domains = self.domain_array[self.current_indices]
            non_target_mask = ~np.isin(current_domains, self.target_domains)
            non_target_positions = np.where(non_target_mask)[0]
            num_non_target = len(non_target_positions)
            drift_samples = int(len(self.current_indices) * self.drift_rate)
            replace_count = min(drift_samples, num_non_target)
            # Skip if no samples to replace
            if replace_count == 0:
                return Subset(self.dataset, self.current_indices.tolist())
            # Select non-target samples to replace
            selected_positions = np.random.choice(non_target_positions, replace_count, replace=False)
            # Sample new target indices
            available_target = np.setdiff1d(self.target_indices, self.current_indices)
            if len(available_target) >= replace_count:
                new_indices = np.random.choice(available_target, replace_count, replace=False)
            else:
                print("Warning: Not enough unique target samples. Sampling with replacement.")
                new_indices = np.random.choice(self.target_indices, replace_count, replace=True)
            # Replace selected
            self.current_indices[selected_positions] = new_indices
            return Subset(self.dataset, self.current_indices.tolist())
