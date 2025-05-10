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
class PACSDomainDrift:
    def __init__(self, data_handler, source_domains, target_domains, drift_rate, desired_size=None):
        self.dataset = data_handler.dataset
        self.source_domains = source_domains
        self.target_domains = target_domains
        self.drift_rate = drift_rate
        self.desired_size = desired_size
        self.domain_array = np.array([domain for _, _, domain in self.dataset])
        self.domain_to_indices = {d: np.where(self.domain_array == d)[0] for d in set(self.domain_array)}
        self.target_indices = np.concatenate([self.domain_to_indices[d] for d in self.target_domains if d in self.domain_to_indices])
        self.current_indices = None
    
    def apply(self):
        
# VLCS Drift

# DomainNet Drift