# models/base_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

############ BASE MODEL ARCHITECTURE CLASS ############
# A class to inherit from for raw neural network architectures
# Used as a way to standardize operation for all different architecture models
# For all neural net architectures, always inherit this class
##################################################
class BaseModelArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError("Architecture classes must implement forward() method")

############ BASE NEURAL NETWORK ############
# A general class for operations with different neural network architectures
# Allows for different common nn-related operations
##################################################
class BaseNeuralNetwork():
    def __init__(self, model_architecture, device=None):
        self.model = model_architecture
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, data_loader, optimizer, loss_fn, epochs, verbose=False):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")
    
    def evaluate(self, data_loader, metric_fn, verbose=False):
        self.model.eval()
        total_metric = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_metric += metric_fn(outputs, targets).item()
        avg_metric = total_metric / len(data_loader)
        if verbose:
            print(f"Evaluation metric: {avg_metric:.4f}")
        return avg_metric
    
    def get_params(self):
        return self.model.state_dict()
    
    def set_params(self, params):
        self.model.load_state_dict(params)
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_model_architecture(self, verbose=True):
        if verbose:
            print(self.model)
        return self.model
    
    def get_layer_names(self, verbose=True):
        names = []
        for name, _ in self.model.named_parameters():
            names.append(name)
            if verbose:
                print(name)
        return names
    
    def get_layer_weights(self, layer_name, verbose=True):
        for name, param in self.model.named_parameters():
            if name == layer_name:
                if verbose:
                    print(f"Weights for layer {name}:")
                    print(param.data)
                return param.data
        if verbose:
            print(f"Layer {layer_name} not found in the model.")
        return None
    
    def get_device(self, verbose=False):
        if verbose:
            print(self.device)
        return self.device
    
    def move_to_device(self, new_device, verbose=False):
        if verbose:
            print(f"New device is: {new_device}")
        self.device = new_device
        self.model.to(self.device)
