# data_operations/data_handler.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Callable, Union, List, Tuple, Any
from abc import ABC, abstractmethod

from data_splitter import *

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
    
# MNIST dataset handler
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
        
# FMNIST dataset handler
class FashionMNISTDataHandler(BaseDataHandler):
    def get_default_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        return transform, transform
    
    def load_data(self, data_dir='./data', download=True, **kwargs):
        if self.transform is None:
            self.transform, test_transform = self.get_default_transforms()
        else:
            test_transform = self.transform
        
        self.dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'Fashion-MNIST',
            'train_size': len(self.dataset),
            'test_size': len(self.test_dataset),
            'num_classes': 10,
            'input_shape': (1, 28, 28)
        }

# CIFAR10 dataset handler
class CIFAR10DataHandler(BaseDataHandler):
    def get_default_transforms(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        return train_transform, test_transform
    
    def load_data(self, data_dir='./data', download=True, **kwargs):
        if self.transform is None:
            self.transform, test_transform = self.get_default_transforms()
        else:
            test_transform = self.transform
        
        self.dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'CIFAR-10',
            'train_size': len(self.dataset),
            'test_size': len(self.test_dataset),
            'num_classes': 10,
            'input_shape': (3, 32, 32)
        }

# CIFAR100 dataset handler
class CIFAR100DataHandler(BaseDataHandler):
    def get_default_transforms(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        return train_transform, test_transform
    
    def load_data(self, data_dir='./data', download=True, **kwargs):
        if self.transform is None:
            self.transform, test_transform = self.get_default_transforms()
        else:
            test_transform = self.transform
        
        self.dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'CIFAR-100',
            'train_size': len(self.dataset),
            'test_size': len(self.test_dataset),
            'num_classes': 100,
            'input_shape': (3, 32, 32)
        }        

if __name__ == "__main__":
    def test_dataset_handler(handler, name, num_clients=5, num_classes_per_client=2):
        print(f"\n=== Testing {name} Handler ===")
        
        # Load data
        handler.load_data(download=True)
        
        # Print dataset info
        info = handler.get_dataset_info()
        print(f"\n{name} Dataset Info:")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test IID splitting
        print(f"\nTesting IID split on {name}:")
        iid_clients = handler.split_data(num_clients=num_clients, strategy='iid')
        for i, dataset in enumerate(iid_clients):
            print(f"Client {i} dataset size: {len(dataset)}")
        
        # Test non-IID splitting
        print(f"\nTesting non-IID split on {name}:")
        non_iid_clients = handler.split_data(
            num_clients=num_clients,
            strategy='non_iid_default',
            num_classes_per_client=num_classes_per_client
        )
        
        for i, dataset in enumerate(non_iid_clients):
            # Get unique labels for this client
            labels = [handler.dataset.targets[j] if isinstance(handler.dataset.targets, list) 
                     else handler.dataset.targets[j].item() 
                     for j in dataset.indices]
            unique_labels = set(labels)
            print(f"Client {i} dataset size: {len(dataset)}, unique classes: {unique_labels}")
        
        # Test data loaders
        train_loader = handler.get_train_loader(batch_size=32)
        test_loader = handler.get_test_loader(batch_size=32)
        print(f"\nNumber of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        
        # Test batch shapes
        train_batch, train_labels = next(iter(train_loader))
        print(f"\n{name} training batch shape: {train_batch.shape}")
        print(f"{name} training labels shape: {train_labels.shape}")

    print("Testing Data Handlers...")
    
    # Test 1: BaseDataHandler with numpy arrays
    print("\n=== Test 1: BaseDataHandler with numpy arrays ===")
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 5, 1000)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randint(0, 5, 200)
    
    handler = BaseDataHandler()
    handler.load_data(
        train_data=X_train, 
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test
    )
    
    print(f"Training dataset size: {len(handler.dataset)}")
    print(f"Test dataset size: {len(handler.test_dataset)}")
    
    # Test all dataset handlers
    test_dataset_handler(MNISTDataHandler(), "MNIST")
    test_dataset_handler(FashionMNISTDataHandler(), "Fashion-MNIST")
    test_dataset_handler(CIFAR10DataHandler(), "CIFAR-10")
    test_dataset_handler(CIFAR100DataHandler(), "CIFAR-100", num_classes_per_client=5)
    
    print("\nAll tests completed successfully!")