# data_operations/data_handler.py

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from .data_splitter import iid_split, non_iid_split, ClientSpec
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Works with any type of data
class BaseDataHandler():
    def __init__(self, transform=None):
        self.train_dataset = None
        self.test_dataset = None
        self.transform = transform
        
    def load_data(self, train_data, train_labels=None, test_data=None, test_labels=None, **kwargs):
        # Handle different types of data
        if isinstance(train_data, Dataset):
            self.train_dataset = train_data
            if test_data is not None and isinstance(test_data, Dataset):
                self.test_dataset = test_data
            
        else:
            if isinstance(train_data, np.ndarray):
                train_data = torch.from_numpy(train_data).float()
            if train_labels is not None and isinstance(train_labels, np.ndarray):
                train_labels = torch.from_numpy(train_labels)
            
            if train_labels is None:
                self.train_dataset = TensorDataset(train_data)
            else:
                self.train_dataset = TensorDataset(train_data, train_labels)
            
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
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")
    
        if strategy == 'iid':
            train_split = iid_split(self.train_dataset, num_clients)
        elif strategy in ['non_iid_default', 'dirichlet', 'pathological']:
            train_split = non_iid_split(self.train_dataset, num_clients, strategy, **kwargs)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        test_split = iid_split(self.test_dataset, num_clients)
        return train_split, test_split
        
    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_train_loader(self, batch_size, shuffle=True):
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded")
        return self.get_dataloader(self.train_dataset, batch_size, shuffle)
    
    def get_test_loader(self, batch_size, shuffle=True):
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded")
        return self.get_dataloader(self.test_dataset, batch_size, shuffle)
    
    def get_client_dataloaders(self, train_dataset, test_dataset, batch_size, 
                             train_shuffle=True, test_shuffle=False):
        train_loader = self.get_dataloader(train_dataset, batch_size, train_shuffle)
        test_loader = self.get_dataloader(test_dataset, batch_size, test_shuffle)
        return train_loader, test_loader
    
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
        
        self.train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.train_dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'MNIST',
            'train_size': len(self.train_dataset),
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
        
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.train_dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'Fashion-MNIST',
            'train_size': len(self.train_dataset),
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
        
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.train_dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'CIFAR-10',
            'train_size': len(self.train_dataset),
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
        
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=download,
            transform=test_transform
        )
        
    def get_dataset_info(self):
        if self.train_dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'CIFAR-100',
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'num_classes': 100,
            'input_shape': (3, 32, 32)
        }        

# PACS dataset and data handler
class PACSDataset(Dataset):
    """PyTorch Dataset wrapper for PACS data"""
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: List of tuples (image, label, domain)
            transform: Optional transform
        """
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label, domain = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, domain

class PACSDataHandler(BaseDataHandler):
    def __init__(self, transform=None, load_data=True):
        super().__init__(transform)
        self.domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        self.categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person',]
        if transform is None:
            self.transform = self.get_default_transforms()
        else:
            self.transform = transform
        # Load data by default
        if load_data:
            self.load_data()
        
    def get_default_transforms(self):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transform
    
    def load_data(self, **kwargs):
        if self.transform is None:
            self.transform = self.get_default_transforms()
        
        # Initialize dataset
        fds = FederatedDataset(
            dataset="flwrlabs/pacs",
            partitioners={"train": IidPartitioner(num_partitions=1)}
        )
        
        # Load single partition which contains all data
        partition = fds.load_partition(0)
        
        # Convert to list of (image, label, domain) tuples
        data_list = [
            (sample['image'], sample['label'], sample['domain']) 
            for sample in partition
        ]
    
        # Create PyTorch datasets
        self.dataset = PACSDataset(data_list, transform=self.transform)
        
    def get_domain_data(self, domain: str) -> tuple:
        if domain not in self.domains:
            raise ValueError(f"Domain must be one of {self.domains}")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")
            
        # Get indices for the specified domain in train set
        indices = [
            i for i in range(len(self.dataset))
            if self.dataset.data[i][2] == domain  # domain is third element in tuple
        ]
        
        return Subset(self.dataset, indices)
    
    def get_dataset_info(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        return {
            'name': 'PACS',
            'dataset_size': len(self.dataset),
            'num_classes': len(self.categories),
            'input_shape': (3, 224, 224),
            'categories': self.categories,
            'domains': self.domains
        }

# VLCS dataset and data handler

# DomainNet dataset and data handler