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
