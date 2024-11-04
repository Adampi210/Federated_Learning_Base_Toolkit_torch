# federated/client.py

import torch
from fl_toolkit.models import BaseNeuralNetwork

class FederatedClient():
    def __init__(self, client_id, model_architecture, device=None):
        self.client_id = client_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaseNeuralNetwork(model_architecture, device=self.device)
        self.data_loader = None 
        
    def set_data(self, data_loader):
        self.data_loader = data_loader
        
    def train(self, epochs, optimizer, loss_fn, verbose=False):
        if self.data_loader is None:
            raise ValueError("Data loader is not set. Use set_data() method to set the data loader")
        self.model.train(self.data_loader, optimizer, loss_fn, epochs, verbose)
    
    def evaluate(self, metric_fn, verbose=False):
        if self.data_loader is None:
            raise ValueError("Data loader is not set. Use set_data() method to set the data loader")
        return self.model.evaluate(self.data_loader, metric_fn, verbose)

    def get_model_params(self):
        return self.model.get_params()
    
    def set_model_params(self, params):
        self.model.set_params(params)
    
    def get_client_id(self):
        return self.client_id
    
