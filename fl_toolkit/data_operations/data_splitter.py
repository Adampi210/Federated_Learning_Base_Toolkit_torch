# data_operations/data_splitter.py

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from collections import defaultdict

# IID split
def iid_split(dataset, num_clients):
    data_len = len(dataset)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    
    base_size = data_len // num_clients
    extra = data_len % num_clients
    
    client_datasets = []
    start_idx = 0
    
    for i in range(num_clients):
        size = base_size + (1 if i < extra else 0)
        client_indices = indices[start_idx:start_idx + size]
        client_datasets.append(Subset(dataset, client_indices))
        start_idx += size
    
    return client_datasets

# Get labels for different types of datasets
def _get_targets(dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    elif hasattr(dataset, 'labels'):
        return dataset.labels
    else:
        try:
            return [dataset[i][1] for i in range(len(dataset))]
        except:
            raise ValueError("Could not extract labels from dataset")

# Default strategy for Non-IID split, selected number of classes per client
def _non_iid_default_split(dataset, num_clients, num_classes_per_client):
    targets = _get_targets(dataset)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    if num_classes_per_client > num_classes:
        raise ValueError(f"num_classes_per_client ({num_classes_per_client}) cannot be greater than total number of classes ({num_classes})")
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    client_datasets = []
    for i in range(num_clients):
        client_classes = np.random.choice(unique_classes, size=num_classes_per_client, replace=False)
        client_indices = []
        
        for class_label in client_classes:
            class_idx = class_indices[class_label]
            num_samples = len(class_idx) // (num_clients * num_classes_per_client // num_classes)
            selected_indices = np.random.choice(class_idx, size=num_samples, replace=False)
            client_indices.extend(selected_indices)
        
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets

# Non-IID split using different strategies
def non_iid_split(dataset, num_clients, strategy='non_iid_default', **kwargs):
    if strategy == 'non_iid_default':
        if 'num_classes_per_client' not in kwargs:
            raise ValueError("num_classes_per_client must be provided for non_iid_default strategy")
        return _non_iid_default_split(dataset, num_clients, kwargs['num_classes_per_client'])
    else:
        raise ValueError(f"Unknown non-IID strategy: {strategy}")

# PACS domain and  class splitting
# Specifies which domains and how many classes a client should get
class ClientSpec:
    def __init__(self, domains, num_classes, min_samples_per_class):
        self.domains = domains
        self.num_classes = num_classes
        self.min_samples_per_class = min_samples_per_class

# Helper to get domains and labels from a given datapoint
def get_domain_and_label(dataset, idx):
    if hasattr(dataset, '__getitem__'):
        _, label, domain = dataset[idx]
        return domain, label
    raise ValueError("Dataset doesn't support item access")

def custom_pacs_split(dataset, client_specs, balance_classes):
    # Group all data by domain and class
    domain_class_indices = defaultdict(lambda: defaultdict(list))
    for idx in range(len(dataset)):
        domain, label = get_domain_and_label(dataset, idx)
        domain_class_indices[domain][label].append(idx)
    
    # Get all unique classes
    all_classes = set()
    for domain_data in domain_class_indices.values():
        all_classes.update(domain_data.keys())
    all_classes = list(all_classes)
    
    client_datasets = []
    
    # For each client specification
    for spec in client_specs:
        # Verify domains exist
        for domain in spec.domains:
            if domain not in domain_class_indices:
                raise ValueError(f"Domain {domain} not found in dataset")
        
        # Find available classes for this client's domains
        available_classes = set()
        class_counts = defaultdict(int)  # Track samples per class across specified domains
        
        for domain in spec.domains:
            for class_label in domain_class_indices[domain]:
                available_classes.add(class_label)
                class_counts[class_label] += len(domain_class_indices[domain][class_label])
        
        # Filter classes that don't meet minimum sample requirement
        if spec.min_samples_per_class:
            available_classes = {
                class_label for class_label in available_classes 
                if class_counts[class_label] >= spec.min_samples_per_class
            }
        
        if len(available_classes) < spec.num_classes:
            raise ValueError(
                f"Not enough classes available for client specification. "
                f"Required: {spec.num_classes}, Available: {len(available_classes)}"
            )
        
        # Randomly select classes for this client
        selected_classes = np.random.choice(
            list(available_classes), 
            size=spec.num_classes, 
            replace=False
        )
        
        # Collect indices for selected classes from specified domains
        client_indices = []
        
        if balance_classes:
            # Find minimum number of samples per class across selected domains
            min_samples = float('inf')
            for class_label in selected_classes:
                total_samples = sum(
                    len(domain_class_indices[domain][class_label])
                    for domain in spec.domains
                )
                min_samples = min(min_samples, total_samples)
                
            # Collect balanced samples
            for class_label in selected_classes:
                class_indices = []
                for domain in spec.domains:
                    if class_label in domain_class_indices[domain]:
                        class_indices.extend(domain_class_indices[domain][class_label])
                
                # Randomly sample to maintain balance
                samples_per_class = min(min_samples, len(class_indices))
                selected = np.random.choice(class_indices, 
                                         size=samples_per_class, 
                                         replace=False)
                client_indices.extend(selected)
        else:
            # Collect all available samples
            for class_label in selected_classes:
                for domain in spec.domains:
                    if class_label in domain_class_indices[domain]:
                        client_indices.extend(domain_class_indices[domain][class_label])
        
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets