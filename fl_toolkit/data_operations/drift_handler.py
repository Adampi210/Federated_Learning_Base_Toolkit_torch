# data_operations/drift_handlers.py

import random
from torch.utils.data import Subset
from torchvision import transforms
import PIL
import torchvision.transforms.functional as F
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

class TransformedSubset(Subset):
    def __init__(self, dataset, indices, drifted_indices, transform):
        super().__init__(dataset, indices)
        self.drifted_indices = drifted_indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        if real_idx in self.drifted_indices:
            x = self.transform(x)
        return x, y

class CIFAR10DomainDrift:
    default_transform = transforms.ColorJitter(brightness=0.5)

    def __init__(self, drift_rate, desired_size=None, transform=None):
        self.drift_rate = drift_rate
        self.desired_size = desired_size
        self.transform = transform if transform else self.default_transform
        self.base_indices = set()
        self.drifted_indices = set()
    
    def apply(self, dataset):
        # First call - initialize with base samples
        if not self.base_indices and not self.drifted_indices:
            all_indices = list(range(len(dataset)))
            initial_size = len(all_indices) if self.desired_size is None else min(len(all_indices), self.desired_size)
            self.base_indices = set(random.sample(all_indices, initial_size))
            return TransformedSubset(dataset, list(self.base_indices), self.drifted_indices, self.transform)
        
        current_size = len(self.base_indices) + len(self.drifted_indices)
        drift_samples = int(current_size * self.drift_rate)
        
        if self.desired_size is None or current_size < self.desired_size:
            # Case 1: Can add more samples
            space_left = float('inf') if self.desired_size is None else self.desired_size - current_size
            drift_samples = min(drift_samples, int(space_left))
            
            to_drift = self._sample_indices(self.base_indices, drift_samples)
            self.base_indices -= set(to_drift)
            self.drifted_indices.update(to_drift)
        else:
            # Case 2: At desired size, replace samples
            to_remove = self._sample_indices(self.drifted_indices, drift_samples)
            self.drifted_indices -= set(to_remove)
            
            to_drift = self._sample_indices(self.base_indices, drift_samples)
            self.base_indices -= set(to_drift)
            self.drifted_indices.update(to_drift)
            self.base_indices.update(to_remove)
            
        return TransformedSubset(dataset, list(self.base_indices | self.drifted_indices), 
                               self.drifted_indices, self.transform)
    
    def _sample_indices(self, available_indices, n):
        if n <= 0:
            return []
        n = min(n, len(available_indices))
        return random.sample(list(available_indices), n)

class CIFAR10DriftTypes:
    @staticmethod
    def gaussian_noise(base_severity=0.1, variation=0.02):
        """
        Noise severity will randomly vary in [base_severity - variation, base_severity + variation].
        """
        def add_noise(img):
            # Sample a random noise level near 'base_severity'
            current_severity = random.uniform(base_severity - variation, base_severity + variation)
            return img + torch.randn_like(img) * current_severity
        
        return transforms.Compose([
            transforms.Lambda(add_noise)
        ])
    
    @staticmethod
    def rotation_drift(base_angle=30, angle_variation=5):
        """
        Rotation angle will randomly vary in [base_angle - angle_variation, base_angle + angle_variation].
        """
        def rotate(img):
            current_angle = random.uniform(base_angle - angle_variation, base_angle + angle_variation)
            return F.rotate(img, current_angle)
        
        return transforms.Compose([
            transforms.Lambda(rotate)
        ])
    
    @staticmethod
    def blur_drift(kernel_size=3, base_sigma=2.0, sigma_variation=0.5):
        """
        Sigma will randomly vary in [base_sigma - sigma_variation, base_sigma + sigma_variation].
        """
        def random_blur(img):
            current_sigma = random.uniform(base_sigma - sigma_variation, base_sigma + sigma_variation)
            return F.gaussian_blur(img, kernel_size=kernel_size, sigma=current_sigma)
        
        return transforms.Compose([
            transforms.Lambda(random_blur)
        ])
    
    @staticmethod
    def color_shift(base_brightness=1.3, brightness_variation=0.1,
                    base_contrast=1.2, contrast_variation=0.1):
        """
        Brightness factor in [base_brightness - brightness_variation, base_brightness + brightness_variation].
        Contrast factor in [base_contrast - contrast_variation, base_contrast + contrast_variation].
        """
        def random_color(img):
            # Random brightness and contrast near their base
            brightness_factor = random.uniform(base_brightness - brightness_variation,
                                               base_brightness + brightness_variation)
            contrast_factor = random.uniform(base_contrast - contrast_variation,
                                             base_contrast + contrast_variation)
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            return img
        
        return transforms.Compose([
            transforms.Lambda(random_color)
        ])
    
    @staticmethod
    def intensity_drift(base_factor=0.8, factor_variation=0.05):
        """
        Intensity factor (multiplicative) will vary in 
        [base_factor - factor_variation, base_factor + factor_variation].
        """
        def random_intensity(img):
            factor = random.uniform(base_factor - factor_variation, base_factor + factor_variation)
            return img * factor
        
        return transforms.Compose([
            transforms.Lambda(random_intensity)
        ])
