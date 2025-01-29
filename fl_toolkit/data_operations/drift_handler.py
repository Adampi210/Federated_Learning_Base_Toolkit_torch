# data_operations/drift_handlers.py

import random
import torch
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import PIL
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

class CIFAR10DomainDrift:
    def __init__(self, drift_rate, transform=None, seed=None):
        self.drift_rate = min(drift_rate, 1.0)
        self.transform = transform
        if seed is not None:
            random.seed(seed)
    
    def apply(self, dataset):
        # Get the original data
        data = []
        targets = []
        for x, y in dataset:
            data.append(x)
            targets.append(y)
        
        # Convert to tensors
        data = torch.stack(data)
        targets = torch.tensor(targets)
        
        # Select indices to drift
        n_samples = len(dataset)
        n_drift = int(n_samples * self.drift_rate)
        drift_indices = random.sample(range(n_samples), n_drift)
        
        # Apply drift to selected samples
        for idx in drift_indices:
            x = data[idx]
            # Unnormalize
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            x = x * std + mean
            # Convert to PIL for transformation
            x = transforms.ToPILImage()(x)
            # Apply drift transform
            x = self.transform(x)
            # Convert back to tensor and normalize
            x = transforms.ToTensor()(x)
            x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
            # Store back
            data[idx] = x
            
        # Create new TensorDataset
        return torch.utils.data.TensorDataset(data, targets)

class CIFAR10DriftTypes:
    @staticmethod
    def color_balance_drift(factor=0.4):
        """Color adjustments with stronger factor"""
        def apply_color_balance(img):
            img = transforms.functional.adjust_saturation(img, 1 + factor)
            img = transforms.functional.adjust_hue(img, factor * 0.5)
            return img
        return apply_color_balance
    
    @staticmethod
    def bounded_darkness_drift(factor=0.2):
        """Reduces brightness with sigmoid-bounded factor to prevent total darkness"""
        def apply_darkness(img):
            # Use sigmoid to bound the darkening effect
            darkness = 1 - (1 / (1 + np.exp(-factor)))  # Will be bounded between 0 and 1
            min_brightness = 0.6  # Never go below 60% brightness
            effective_factor = min_brightness + (1 - min_brightness) * darkness
            return transforms.functional.adjust_brightness(img, effective_factor)
        return apply_darkness
    
    @staticmethod
    def soft_gamma_drift(gamma_factor=0.3):
        """Applies gamma with bounded range to avoid extreme darkening"""
        def apply_soft_gamma(img):
            # Bound gamma between 1 and 2.5 using sigmoid
            max_gamma = 1.3
            min_gamma = 1.05
            bounded_factor = min_gamma + (max_gamma - min_gamma) * (1 / (1 + np.exp(-gamma_factor)))
            return transforms.functional.adjust_gamma(img, bounded_factor)
        return apply_soft_gamma
    
    @staticmethod
    def color_wash_drift(factor=0.3):
        """Gradually washes out colors while preserving image structure"""
        def apply_color_wash(img):
            # Reduce saturation and slightly adjust brightness
            img = transforms.functional.adjust_saturation(img, 1 - factor * 0.5)  # Reduce saturation
            # Add slight brightness adjustment but keep it above 0.7
            brightness_factor = 0.7 + 0.3 * (1 / (1 + np.exp(-factor)))
            img = transforms.functional.adjust_brightness(img, brightness_factor)
            return img
        return apply_color_wash

    @staticmethod
    def gaussian_noise(base_severity=0.5):
        """
        Apply Gaussian noise with specified severity
        """
        def add_noise(img):
            img = torch.tensor(np.array(img))
            noise = torch.randn_like(img.float()) * base_severity
            noisy_img = img.float() + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            return transforms.ToPILImage()(noisy_img)
        return add_noise

    @staticmethod
    def rotation_drift(base_angle=30):
        """
        Fixed rotation by base_angle degrees
        """
        return lambda img: transforms.functional.rotate(img, base_angle)

    @staticmethod
    def blur_drift(kernel_size=5, sigma=2.0):
        """
        Fixed Gaussian blur
        """
        return lambda img: transforms.functional.gaussian_blur(img, kernel_size, sigma)
    @staticmethod
    def gaussian_noise(base_severity=10, variation=1):
        """
        Note: Since torchvision doesn't have a built-in Gaussian noise transform,
        we'll use RandomErasing as the closest alternative for introducing random noise.
        """
        return transforms.Compose([
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.05), ratio=(0.3, 3.3), 
                          value='random')
        ])

    @staticmethod
    def rotation_drift(base_angle=30, angle_variation=5):
        """
        Rotation angle will randomly vary in [base_angle - angle_variation, base_angle + angle_variation].
        """
        return transforms.Compose([
            transforms.RandomRotation(
                degrees=(base_angle - angle_variation, base_angle + angle_variation)
            )
        ])

    @staticmethod
    def blur_drift(kernel_size=3, base_sigma=2.0, sigma_variation=0.5):
        """
        Applies Gaussian blur with specified kernel size and sigma range.
        """
        return transforms.Compose([
            transforms.GaussianBlur(
                kernel_size=kernel_size,
                sigma=(base_sigma - sigma_variation, base_sigma + sigma_variation)
            )
        ])

    @staticmethod
    def color_shift(base_brightness=1.3, brightness_variation=0.1,
                    base_contrast=1.2, contrast_variation=0.1):
        """
        Applies random brightness and contrast adjustments.
        """
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=(base_brightness - brightness_variation,
                          base_brightness + brightness_variation),
                contrast=(base_contrast - contrast_variation,
                         base_contrast + contrast_variation)
            )
        ])

    @staticmethod
    def intensity_drift(base_factor=0.8, factor_variation=0.05):
        """
        Note: Using ColorJitter's brightness adjustment as the closest equivalent
        to intensity scaling in torchvision.transforms
        """
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=(base_factor - factor_variation,
                          base_factor + factor_variation)
            )
        ])