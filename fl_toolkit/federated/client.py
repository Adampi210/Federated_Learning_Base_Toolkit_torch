# federated/client.py

import torch
from torch.utils.data import DataLoader
from fl_toolkit.models import BaseNeuralNetwork
from fl_toolkit.data_operations import DriftedDataset

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
    def __init__(self,
                 client_id,
                 model_architecture,
                 data_drift=None,
                 concept_drift=None,
                 device=None):
        """
        Args:
            client_id
            model_architecture
            data_drift: DataDrift instance for applying data drift
            concept_drift: ConceptDrift instance for applying concept drift
            device
        """
        super().__init__(client_id, model_architecture, device)
        self.data_drift = data_drift
        self.concept_drift = concept_drift
        self.original_train_loader = None
        self.original_test_loader = None

    # Applies current drift configurations to the data
    def apply_drift(self):
        if self.train_loader is not None:
            drifted_train_dataset = DriftedDataset(
                dataset=self.train_loader.dataset,
                data_drift=self.data_drift,
                concept_drift=self.concept_drift
            )
            self.train_loader = DataLoader(
                drifted_train_dataset,
                batch_size=self.train_loader.batch_size,
                shuffle=self.train_loader.shuffle,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory
            )
           
        if self.test_loader is not None:
            drifted_test_dataset = DriftedDataset(
                dataset=self.test_loader.dataset,
                data_drift=self.data_drift,
                concept_drift=self.concept_drift
            )
            self.test_loader = DataLoader(
                drifted_test_dataset,
                batch_size=self.test_loader.batch_size,
                num_workers=self.test_loader.num_workers,
                pin_memory=self.test_loader.pin_memory
            )
    
    # Updates current drift configurations, applies new drift to the data
    def update_drift(self,
                    data_drift=None,
                    concept_drift=None):
        if data_drift is not None:
            self.data_drift = data_drift
        if concept_drift is not None:
            self.concept_drift = concept_drift
        self.apply_drift()
        
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