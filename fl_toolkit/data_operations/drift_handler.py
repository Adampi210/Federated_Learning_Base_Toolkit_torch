# data_operations/drift_handlers.py

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import random
from typing import Callable, Optional, Union, List, Tuple
from collections import defaultdict
from enum import Enum

class DataDriftType(Enum):
    INJECTION = "injection"
    IMBALANCE = "imbalance"
    CLASS_SWAP = "class_swap"
    FEATURE_NOISE = "feature_noise"

class ConceptDriftType(Enum):
    ROTATION = "rotation"
    SCALING = "scaling"
    NOISE = "noise"
    BLUR = "blur"
    COLOR_JITTER = "color_jitter"

# Data Drift Handler
class DataDrift:
    def __init__(self, 
                 drift_type: Union[str, DataDriftType],
                 **kwargs):
        """        
        Args:
            drift_type: Type of data drift to apply, see below for data drift types
            **kwargs: Config parameters for the specific data drift type
                INJECTION:
                    - injection_ratio: float (0-1)
                    - source_classes: List[int]
                    - target_classes: List[int]
                IMBALANCE:
                    - target_classes: List[int]
                    - reduction_ratio: float (0-1)
                CLASS_SWAP:
                    - swap_pairs: List[Tuple[int, int]]
                    - swap_ratio: float (0-1)
                FEATURE_NOISE:
                    - noise_std: float
                    - feature_indices: List[int]
        """
        if isinstance(drift_type, str):
            drift_type = DataDriftType(drift_type.lower())
        self.drift_type = drift_type
        self.config = kwargs
        self._validate_config()
        
    def _validate_config(self):
        if self.drift_type == DataDriftType.INJECTION:
            required = {'injection_ratio', 'source_classes', 'target_classes'}
            if not all(k in self.config for k in required):
                raise ValueError(f"INJECTION drift requires: {required}")
            if not 0 <= self.config['injection_ratio'] <= 1:
                raise ValueError("injection_ratio must be between 0 and 1")
                
        elif self.drift_type == DataDriftType.IMBALANCE:
            required = {'target_classes', 'reduction_ratio'}
            if not all(k in self.config for k in required):
                raise ValueError(f"IMBALANCE drift requires: {required}")
            if not 0 <= self.config['reduction_ratio'] <= 1:
                raise ValueError("reduction_ratio must be between 0 and 1")
                
        elif self.drift_type == DataDriftType.CLASS_SWAP:
            required = {'swap_pairs', 'swap_ratio'}
            if not all(k in self.config for k in required):
                raise ValueError(f"CLASS_SWAP drift requires: {required}")
            if not 0 <= self.config['swap_ratio'] <= 1:
                raise ValueError("swap_ratio must be between 0 and 1")
                
        elif self.drift_type == DataDriftType.FEATURE_NOISE:
            required = {'noise_std', 'feature_indices'}
            if not all(k in self.config for k in required):
                raise ValueError(f"FEATURE_NOISE drift requires: {required}")
            if self.config['noise_std'] < 0:
                raise ValueError("noise_std must be non-negative")

    def apply(self, dataset: Dataset) -> Dataset:
        if self.drift_type == DataDriftType.INJECTION:
            return self._apply_injection(dataset)
        elif self.drift_type == DataDriftType.IMBALANCE:
            return self._apply_imbalance(dataset)
        elif self.drift_type == DataDriftType.CLASS_SWAP:
            return self._apply_class_swap(dataset)
        elif self.drift_type == DataDriftType.FEATURE_NOISE:
            return self._apply_feature_noise(dataset)
        else:
            raise ValueError(f"Unknown drift type: {self.drift_type}")

    # Inject samples from source classes into target classes
    def _apply_injection(self, dataset: Dataset) -> Dataset:
        if not hasattr(dataset, 'targets'):
            raise ValueError("Dataset must have 'targets' attribute")
            
        targets = torch.tensor(dataset.targets)
        indices = torch.arange(len(dataset))
        
        source_indices = torch.cat([indices[targets == c] 
                                  for c in self.config['source_classes']])
        target_indices = torch.cat([indices[targets == c] 
                                  for c in self.config['target_classes']])
        
        n_inject = int(len(target_indices) * self.config['injection_ratio'])
        inject_from = source_indices[torch.randperm(len(source_indices))[:n_inject]]
        inject_to = target_indices[torch.randperm(len(target_indices))[:n_inject]]
        
        index_mapping = {int(to): int(from_) for to, from_ in zip(inject_to, inject_from)}
        
        return DriftedDataset(dataset, index_mapping=index_mapping)

    # Reduce samples from target classes to create imbalance
    def _apply_imbalance(self, dataset: Dataset) -> Dataset:
        if not hasattr(dataset, 'targets'):
            raise ValueError("Dataset must have 'targets' attribute")
            
        targets = torch.tensor(dataset.targets)
        indices = torch.arange(len(dataset))
        
        target_indices = torch.cat([indices[targets == c] 
                                  for c in self.config['target_classes']])
        n_keep = int(len(target_indices) * (1 - self.config['reduction_ratio']))
        keep_indices = target_indices[torch.randperm(len(target_indices))[:n_keep]]
        
        non_target_indices = torch.cat([indices[targets == c] 
                                      for c in range(max(targets) + 1) 
                                      if c not in self.config['target_classes']])
        
        final_indices = torch.cat([keep_indices, non_target_indices])
        return Subset(dataset, final_indices)

    # Swap samples between pairs of classes
    def _apply_class_swap(self, dataset: Dataset) -> Dataset:
        if not hasattr(dataset, 'targets'):
            raise ValueError("Dataset must have 'targets' attribute")
            
        targets = torch.tensor(dataset.targets)
        indices = torch.arange(len(dataset))
        index_mapping = {}
        
        for class1, class2 in self.config['swap_pairs']:
            class1_indices = indices[targets == class1]
            class2_indices = indices[targets == class2]
            n_swap = int(min(len(class1_indices), len(class2_indices)) 
                        * self.config['swap_ratio'])
            
            swap_from1 = class1_indices[torch.randperm(len(class1_indices))[:n_swap]]
            swap_from2 = class2_indices[torch.randperm(len(class2_indices))[:n_swap]]
            
            for idx1, idx2 in zip(swap_from1, swap_from2):
                index_mapping[int(idx1)] = int(idx2)
                index_mapping[int(idx2)] = int(idx1)
                
        return DriftedDataset(dataset, index_mapping=index_mapping)

    def _apply_feature_noise(self, dataset: Dataset) -> Dataset:
        """Add noise to specific features."""
        if not hasattr(dataset, 'data'):
            raise ValueError("Dataset must have 'data' attribute")
            
        noisy_data = dataset.data.clone()
        noise = torch.randn_like(noisy_data.float()) * self.config['noise_std']
        
        for idx in self.config['feature_indices']:
            noisy_data[..., idx] += noise[..., idx]
            
        dataset.data = noisy_data
        return dataset

# Concept Drift Handler
class ConceptDrift:
    def __init__(self, 
                 drift_type: Union[str, ConceptDriftType],
                 **kwargs):
        """
        Args:
            drift_type: Type of drift to apply
            **kwargs: Configuration parameters for the specific drift type
                ROTATION:
                    - angle_range: Tuple[float, float] (min_angle, max_angle)
                SCALING:
                    - scale_range: Tuple[float, float] (min_scale, max_scale)
                NOISE:
                    - noise_std: float
                BLUR:
                    - kernel_size: int
                    - sigma: float
                COLOR_JITTER:
                    - brightness: float
                    - contrast: float
                    - saturation: float
                    - hue: float
        """
        if isinstance(drift_type, str):
            drift_type = ConceptDriftType(drift_type.lower())
        self.drift_type = drift_type
        self.config = kwargs
        self._validate_config()
        self._setup_transform()
        
    def _validate_config(self):
        if self.drift_type == ConceptDriftType.ROTATION:
            if 'angle_range' not in self.config:
                raise ValueError("ROTATION drift requires 'angle_range'")
            min_angle, max_angle = self.config['angle_range']
            if min_angle > max_angle:
                raise ValueError("min_angle must be less than max_angle")
                
        elif self.drift_type == ConceptDriftType.SCALING:
            if 'scale_range' not in self.config:
                raise ValueError("SCALING drift requires 'scale_range'")
            min_scale, max_scale = self.config['scale_range']
            if min_scale > max_scale or min_scale <= 0:
                raise ValueError("Invalid scale range")
                
        elif self.drift_type == ConceptDriftType.NOISE:
            if 'noise_std' not in self.config:
                raise ValueError("NOISE drift requires 'noise_std'")
            if self.config['noise_std'] < 0:
                raise ValueError("noise_std must be non-negative")
                
        elif self.drift_type == ConceptDriftType.BLUR:
            required = {'kernel_size', 'sigma'}
            if not all(k in self.config for k in required):
                raise ValueError(f"BLUR drift requires: {required}")
            if self.config['kernel_size'] % 2 != 1:
                raise ValueError("kernel_size must be odd")
                
        elif self.drift_type == ConceptDriftType.COLOR_JITTER:
            required = {'brightness', 'contrast', 'saturation', 'hue'}
            if not all(k in self.config for k in required):
                raise ValueError(f"COLOR_JITTER drift requires: {required}")

    def _setup_transform(self):
        if self.drift_type == ConceptDriftType.ROTATION:
            min_angle, max_angle = self.config['angle_range']
            self.transform = lambda x: transforms.functional.rotate(
                x, random.uniform(min_angle, max_angle))
                
        elif self.drift_type == ConceptDriftType.SCALING:
            min_scale, max_scale = self.config['scale_range']
            def scale_fn(x):
                scale = random.uniform(min_scale, max_scale)
                return transforms.functional.resize(
                    x, [int(s * scale) for s in x.shape[-2:]])
            self.transform = scale_fn
            
        elif self.drift_type == ConceptDriftType.NOISE:
            noise_std = self.config['noise_std']
            self.transform = lambda x: x + torch.randn_like(x) * noise_std
            
        elif self.drift_type == ConceptDriftType.BLUR:
            kernel_size = self.config['kernel_size']
            sigma = self.config['sigma']
            self.transform = lambda x: transforms.functional.gaussian_blur(
                x, kernel_size, sigma)
                
        elif self.drift_type == ConceptDriftType.COLOR_JITTER:
            self.transform = transforms.ColorJitter(
                brightness=self.config['brightness'],
                contrast=self.config['contrast'],
                saturation=self.config['saturation'],
                hue=self.config['hue']
            )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

# Dataset wrapper that applies data and/or concept drift handlers
class DriftedDataset(Dataset):
    def __init__(self, 
                 dataset: Dataset,
                 data_drift: Optional[DataDrift] = None,
                 concept_drift: Optional[ConceptDrift] = None,
                 index_mapping: Optional[dict] = None):
        """
        Args:
            dataset: Original dataset
            data_drift: DataDrift instance
            concept_drift: ConceptDrift instance
            index_mapping: Optional mapping for indices (used by some data drifts)
        """
        self.dataset = dataset
        self.data_drift = data_drift
        self.concept_drift = concept_drift
        self.index_mapping = index_mapping or {}
        
    def __getitem__(self, idx):
        # Apply data drift through index mapping if it exists
        mapped_idx = self.index_mapping.get(idx, idx)
        x, y = self.dataset[mapped_idx]
        
        # Apply concept drift if it exists
        if self.concept_drift is not None:
            x = self.concept_drift.apply(x)
            
        return x, y
        
    def __len__(self):
        return len(self.dataset)