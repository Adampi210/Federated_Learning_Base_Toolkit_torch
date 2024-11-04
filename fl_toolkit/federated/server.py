# federated/server.py

import torch
from fl_toolkit.models import BaseNeuralNetwork

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
        
        for name, param in global_params.items():
            weighted_sum = torch.zeros_like(param)
            for client_id, weight in client_weights.items():
                client_params = self.clients[client_id].get_model_params()
                weighted_sum += weight * client_params[name]
            global_params[name] = weighted_sum
        
        self.global_model.set_params(global_params)
        
    def train_round_clients(self, epochs, optimizer, loss_fn, verbose=False):
        for client_id, client in self.clients.items():
            if verbose:
                print(f'Client {client_id} training running')
            client.train(epochs, optimizer, loss_fn, verbose)
            
    def evaluate_global_model(self, data_loader, metric_fn, verbose=False):
        return self.global_model.evaluate(data_loader, metric_fn, verbose)
