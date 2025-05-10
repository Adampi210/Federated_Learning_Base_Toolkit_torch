# federated/client.py

import torch
from torch.utils.data import DataLoader
from fl_toolkit.models import BaseNeuralNetwork
from torch.utils.data import Dataset, Subset
import random 

# For transforming the dataset
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, domain = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, domain

# Basic Federated Learning client class
class FederatedClient():
    def __init__(self, client_id, model_architecture, device=None):
        self.client_id = client_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_architecture is not None:
            self.model = BaseNeuralNetwork(model_architecture, device=self.device)
        self.train_loader = None 
        self.test_loader = None
        
    def set_data(self, train_loader, test_loader=None):
        self.train_loader = train_loader
        if test_loader is not None:
            self.test_loader = test_loader
        
    def train(self, epochs, optimizer, loss_fn, verbose=False):
        if self.train_loader is None:
            raise ValueError("Train loader is not set. Use set_data() method to set the train loader")
        return self.model.train(self.train_loader, optimizer, loss_fn, epochs, verbose)
    
    def update_steps(self, num_updates, optimizer, loss_fn, verbose=False):
        if self.train_loader is None:
            raise ValueError("Train loader is not set. Use set_data() method to set the train loader")
        return self.model.update_steps(self.train_loader, optimizer, loss_fn, num_updates, verbose)
    
    def evaluate(self, metric_fn, verbose=False):
        if self.test_loader is None:
            raise ValueError("Test loader is not set. Use set_data() method to set the test loader")
        return self.model.evaluate(self.test_loader, metric_fn, verbose)

    def get_model(self):
        return self.model.model

    def get_model_params(self):
        return self.model.get_params()
    
    def set_model_params(self, params):
        self.model.set_params(params)
    
    def randomize_weights(self):
        self.model.randomize_weights()
    
    def get_client_id(self):
        return self.client_id
    
# Federated Learning client that experiences concept drift
class FederatedDriftClient(FederatedClient):    
    def __init__(self, client_id, model_architecture, train_domain_drift=None, test_domain_drift=None, device=None):
        super().__init__(client_id, model_architecture, device)
        self.train_domain_drift = train_domain_drift
        self.test_domain_drift = test_domain_drift
        self.original_train_loader = None
        self.original_test_loader = None
        
    def set_data(self, train_loader, test_loader=None):
        self.original_train_loader = train_loader
        self.original_test_loader = test_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if self.train_domain_drift is not None:
            self.apply_train_drift()
        
        if self.test_domain_drift is not None:
            self.apply_test_drift()
        
    def apply_train_drift(self, verbose=False):
        if self.train_loader is not None and self.train_domain_drift is not None:
            drifted_train = self.train_domain_drift.apply(self.original_train_loader.dataset)
            # Apply drift only to the training dataset
            if verbose:
                print('train drift')
                # print(f'photo samples: {self.count_domain_samples(drifted_train, 'photo')}')
                # print(f'sketch samples: {self.count_domain_samples(drifted_train, 'sketch')}')
                # print(f'art painting samples: {self.count_domain_samples(drifted_train, 'art_painting')}')
                # print(f'cartoon samples: {self.count_domain_samples(drifted_train, 'cartoon')}')

            self.train_loader = DataLoader(
                drifted_train,
                batch_size=self.original_train_loader.batch_size,
                shuffle=True,
                num_workers=self.original_train_loader.num_workers
            )

    def apply_test_drift(self, verbose=False):
        if self.test_loader is not None and self.test_domain_drift is not None:
            # Apply drift only to the testing dataset
            drifted_test = self.test_domain_drift.apply(self.original_test_loader.dataset)
            if verbose:
                print('test drift')
                # rint(f'photo samples: {self.count_domain_samples(drifted_test, 'photo')}')
                # print(f'sketch samples: {self.count_domain_samples(drifted_test, 'sketch')}')
                # print(f'art painting samples: {self.count_domain_samples(drifted_test, 'art_painting')}')
                # print(f'cartoon samples: {self.count_domain_samples(drifted_test, 'cartoon')}')

            self.test_loader = DataLoader(
                drifted_test,
                batch_size=self.original_test_loader.batch_size,
                shuffle=False,
                num_workers=self.original_test_loader.num_workers
            )
            
    def count_domain_samples(self, dataset, domain):
        return len([i for i in range(len(dataset)) if dataset[i][2] == domain])

    def get_train_metric(self, metric_fn, verbose = False):
        if self.train_loader is None:
            raise ValueError("Train loader is not set. Use set_data() method to set the train loader")
        return self.model.evaluate(self.train_loader, metric_fn, verbose)

# Agent class that experiences domain drift
class DriftAgent(FederatedClient):
    def __init__(self, client_id, model_architecture, domain_drift, batch_size=32, device=None):
        super().__init__(client_id, model_architecture, device)
        if domain_drift is None:
            raise ValueError("Domain drift cannot be None")
        self.domain_drift = domain_drift
        self.original_dataset = self.domain_drift.dataset
        self.batch_size = batch_size
        self.set_data()

    def set_data(self):
        """Set the dataset for the agent using the initial domains"""
        if self.domain_drift is not None:
            self.current_dataset = self.domain_drift.apply()
        else:
            raise ValueError("Domain drift is not set")
        self.data_loader = DataLoader(self.current_dataset, batch_size=self.batch_size, shuffle=True)
    
    def apply_drift(self):
        """Apply drift to the dataset"""
        """NOTE: This is equivalent to set_data(), but is created for clarity in training"""
        if self.domain_drift is not None:
            self.current_dataset = self.domain_drift.apply()
            self.data_loader = DataLoader(self.current_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            raise ValueError("Domain drift is not set")
        
    def set_drift_rate(self, new_rate):
        """Set the drift rate for the domain drift mechanism."""
        if self.domain_drift is not None:
            self.domain_drift.drift_rate = new_rate
        else:
            raise ValueError("Domain drift is not set")
        
    def set_target_domains(self, new_target_domains):
        """Update the target domains for the drift mechanism."""
        if self.domain_drift is not None:
            self.domain_drift.set_target_domains(new_target_domains)
        else:
            raise ValueError("No domain drift mechanism is set.")
    
    def len_domain_samples(self, domain):
        """Count samples from a specific domain in the current dataset."""
        if self.current_dataset is None:
            raise ValueError("Dataset is not set.")
        return len([i for i in range(len(self.current_dataset)) if self.current_dataset[i][2] == domain])
    
    def get_test_loader(self, test_size=0.1, test_batch_size=32):
        """Create a test loader by randomly sampling from the current dataset."""
        if self.current_dataset is None:
            raise ValueError("Dataset is not set.")
        total_samples = len(self.current_dataset)
        test_samples = int(total_samples * test_size)
        indices = list(range(total_samples))
        test_indices = random.sample(indices, test_samples)
        test_subset = Subset(self.current_dataset, test_indices)
        self.test_subset = test_subset
        return DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

    def train(self, epochs, optimizer, loss_fn, verbose=False):
        if self.train_loader is None:
            raise ValueError("Train loader is not set. Use set_data() method to set the train loader")
        return self.model.train(self.data_loader, optimizer, loss_fn, epochs, verbose)
    
    def update_steps(self, num_updates, optimizer, loss_fn, verbose=False):
        if self.data_loader is None:
            raise ValueError("Train loader is not set. Use set_data() method to set the train loader")
        return self.model.update_steps(self.data_loader, optimizer, loss_fn, num_updates, verbose)
    
    def evaluate(self, metric_fn, verbose=False):
        return self.model.evaluate(self.get_test_loader(), metric_fn, verbose)

# Federated Learning client that utilizes compression algorithms during communication
class FederatedCompressedClient(FederatedClient):
    def __init__(self, client_id, model_architecture, compressor=None, device=None):
        super().__init__(client_id, model_architecture, device)
        self.compressor = compressor
        self.current_masks = None
    
    def get_model_params(self):
        params = self.model.get_params()
        if self.compressor is not None:
            compressed_params, masks = self.compressor.compress(params)
            self.current_masks = masks
            return compressed_params, masks
        return params, None
    
    def set_model_params(self, params):
        current_params = self.model.get_params()
        
        if self.compressor is not None and self.current_masks is not None:
            updated_params = {}
            for name, param in current_params.items():
                if name in self.current_masks:
                    # Create a mask for where parameters should be updated (mask == 1)
                    mask = self.current_masks[name]
                    
                    # Keep old parameters where mask is 0, use new parameters where mask is 1
                    updated_params[name] = (param * (1 - mask)) + (params[name] * mask)
                else:
                    # For parameters without masks, keep original values
                    updated_params[name] = param
                    
            self.model.set_params(updated_params)
        else:
            self.model.set_params(params)