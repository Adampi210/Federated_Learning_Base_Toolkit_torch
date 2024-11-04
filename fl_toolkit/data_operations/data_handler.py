# data_operations/data_handler.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Callable, Union, List, Tuple, Any
from abc import ABC, abstractmethod

# Works with any type of data
class BaseDataHandler():
    def __init__(self, transform=None):
        self.dataset = None
        self.test_dataset = None
        self.transform = transform
        
    def load_data(self, train_data, train_labels=None, test_data=None, test_labels=None, **kwargs):
        # Handle different types of data
        if isinstance(train_data, Dataset):
            self.dataset = train_data
            if test_data is not None and isinstance(test_data, Dataset):
                self.test_dataset = test_data
            
        else:
            if isinstance(train_data, np.ndarray):
                train_data = torch.from_numpy(train_data).float()
            if train_labels is not None and isinstance(train_labels, np.ndarray):
                train_labels = torch.from_numpy(train_labels)
            
            if train_labels is None:
                self.dataset = TensorDataset(train_data)
            else:
                self.dataset = TensorDataset(train_data, train_labels)
            
            if test_data is not None:
                if isinstance(test_data, np.ndarray):
                    test_data = torch.from_numpy(test_data).float()
                if test_labels is not None and isinstance(test_labels, np.ndarray):
                    test_labels = torch.from_numpy(test_labels)
                
                if test_labels is None:
                    self.test_dataset = TensorDataset(test_data)
                else:
                    self.test_dataset = TensorDataset(test_data, test_labels)
    
    def split_data(self, num_clients, strategy="iid", **kwargs):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")
    
        if strategy == 'iid':
            return iid_split(self.dataset, num_clients)
        elif strategy in ['non_iid_default', 'dirichlet', 'pathological']:
            return non_iid_split(self.dataset, num_clients, strategy, **kwargs)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
    
    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_train_loader(self, batch_size, shuffle=True):
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        return self.get_dataloader(self.dataset, batch_size, shuffle)
    
    def get_test_loader(self, batch_size, shuffle=True):
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded")
        return self.get_dataloader(self.test_dataset, batch_size, shuffle)
    
# For managing MNIST dataset 
class MNISTDataHandler(BaseDataHandler):
    def get_default_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform, transform
    
    def load_data(self, data_dir='./data', download=True, **kwargs):
        if self.transform is None:
            self.transform, test_transform = self.get_default_transforms()
        else:
            test_transform = self.transform
        
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'MNIST',
            'train_size': len(self.dataset),
            'test_size': len(self.test_dataset),
            'num_classes': 10,
            'input_shape': (1, 28, 28)
        }