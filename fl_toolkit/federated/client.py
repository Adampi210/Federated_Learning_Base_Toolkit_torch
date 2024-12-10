# federated/client.py

import torch
from torch.utils.data import DataLoader
from fl_toolkit.models import BaseNeuralNetwork
from torch.utils.data import Dataset, Subset
import random 

# Basic Federated Learning client class
class FederatedClient():
    def __init__(self, client_id, model_architecture, device=None):
        self.client_id = client_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.model.train(self.train_loader, optimizer, loss_fn, epochs, verbose)
    
    def evaluate(self, metric_fn, verbose=False):
        if self.test_loader is None:
            raise ValueError("Test loader is not set. Use set_data() method to set the test loader")
        return self.model.evaluate(self.test_loader, metric_fn, verbose)

    def get_model_params(self):
        return self.model.get_params()
    
    def set_model_params(self, params):
        self.model.set_params(params)
    
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
        
    def apply_train_drift(self):
        if self.train_loader is not None and self.train_domain_drift is not None:
            # Apply drift only to the training dataset
            print('train drift')
            drifted_train = self.train_domain_drift.apply(self.original_train_loader.dataset)
            print(f'photo samples: {self.count_domain_samples(drifted_train, 'photo')}')
            print(f'sketch samples: {self.count_domain_samples(drifted_train, 'sketch')}')
            print(f'art painting samples: {self.count_domain_samples(drifted_train, 'art_painting')}')
            print(f'cartoon samples: {self.count_domain_samples(drifted_train, 'cartoon')}')

            self.train_loader = DataLoader(
                drifted_train,
                batch_size=self.original_train_loader.batch_size,
                shuffle=True,
            )

    def apply_test_drift(self):
        if self.test_loader is not None and self.test_domain_drift is not None:
            # Apply drift only to the testing dataset
            drifted_test = self.test_domain_drift.apply(self.original_test_loader.dataset)
            print('test drift')
            print(f'photo samples: {self.count_domain_samples(drifted_test, 'photo')}')
            print(f'sketch samples: {self.count_domain_samples(drifted_test, 'sketch')}')
            print(f'art painting samples: {self.count_domain_samples(drifted_test, 'art_painting')}')
            print(f'cartoon samples: {self.count_domain_samples(drifted_test, 'cartoon')}')

            self.test_loader = DataLoader(
                drifted_test,
                batch_size=self.original_test_loader.batch_size,
                shuffle=False,
            )
    def count_domain_samples(self, dataset, domain):
        return len([i for i in range(len(dataset)) if dataset[i][2] == domain])

    
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