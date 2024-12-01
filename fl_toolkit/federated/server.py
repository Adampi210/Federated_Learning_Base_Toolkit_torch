# federated/server.py

import torch
from fl_toolkit.models import BaseNeuralNetwork
from fl_toolkit.model_operations import *

class FederatedServer():
    def __init__(self, model_architecture, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = BaseNeuralNetwork(model_architecture, device=self.device)
        self.clients = {}
        
    def add_client(self, client):
        self.clients[client.get_client_id()] = client

    def remove_client(self, client_id):
        if client_id in self.clients:
            del self.clients[client_id]
    
    def get_client_ids(self):
        return list(self.clients.keys())        
    
    def get_clients(self):
        return list(self.clients.values())
    
    def distribute_model(self):
        global_params = self.global_model.get_params()
        for client in self.clients.values():
            client.set_model_params(global_params)

    def aggregate_models_FedAVG(self, client_weights=None):
        if not self.clients:
            raise ValueError("No clients available for aggregation")
        
        if client_weights is None:
            client_weights = {client_id: 1 / len(self.clients) for client_id in self.clients}
        
        global_params = self.global_model.get_params()
        first_client_params = next(iter(self.clients.values())).get_model_params()
        
        for name, param in global_params.items():
            param_type = first_client_params[name].dtype
            device = param.device
            
            weighted_sum = torch.zeros_like(first_client_params[name])
            
            for client_id, weight in client_weights.items():
                client_params = self.clients[client_id].get_model_params()
                weight_tensor = torch.tensor(weight, dtype=param_type, device=device)
                weighted_sum += weight_tensor * client_params[name]
            
            global_params[name] = weighted_sum
        
        self.global_model.set_params(global_params)  
       
    def train_round_clients(self, epochs, optimizer, loss_fn, verbose=False):
        for client_id, client in self.clients.items():
            if verbose:
                print(f'Client {client_id} training running')
            client.train(epochs, optimizer, loss_fn, verbose)
            
    def evaluate_global_model(self, data_loader, metric_fn, verbose=False):
        return self.global_model.evaluate(data_loader, metric_fn, verbose)

class FederatedCompressedServer(FederatedServer):
    def __init__(self, model_architecture, device=None):
        super().__init__(model_architecture, device)
    
    def aggregate_models_FedAVG(self, weights=None):
        if not self.clients:
            raise ValueError("No clients available for aggregation")
        
        global_params = self.global_model.get_params()
        
        # Initialize parameter sums and counts
        param_sums = {name: torch.zeros_like(param) for name, param in global_params.items()}
        participation_counts = {name: torch.zeros_like(param) for name, param in global_params.items()}
        
        # Sum up parameters based on masks
        for client in self.clients.values():
            params, masks = client.get_model_params()
            
            for name, param in params.items():
                if masks and name in masks:
                    param_sums[name] += param * masks[name]
                    participation_counts[name] += masks[name]
        
        # Average parameters where at least one client participated
        for name in global_params:
            # Where participation_count > 0, compute average
            # Where participation_count = 0, keep previous value
            mask = (participation_counts[name] > 0).float()
            global_params[name] = (param_sums[name] / (participation_counts[name] + 1e-10)) * mask + \
                                global_params[name] * (1 - mask)
        
        self.global_model.set_params(global_params)
    
    def distribute_model(self):
        """Original distribution method is sufficient since clients handle masking"""
        global_params = self.global_model.get_params()
        for client in self.clients.values():
            client.set_model_params(global_params)