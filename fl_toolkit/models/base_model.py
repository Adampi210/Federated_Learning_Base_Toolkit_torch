# models/base_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np

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
        # If class, create instance
        if isinstance(model_architecture, type):
            self.model = model_architecture()
        # Otherwise use the instance
        else:
            self.model = model_architecture
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, data_loader, optimizer, loss_fn, epochs, verbose=False):
        self.model.train()
        avg_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                if len(batch) > 2:
                    inputs, targets, _ = batch  # Ignore rest
                else:
                    inputs, targets = batch
                # Handle dictionary inputs for transformers
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                # Use dictionary unpacking for model call
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")
            avg_loss += total_loss / len(data_loader)
        avg_loss /= epochs
        return avg_loss
    
    def update_steps(self, data_loader, optimizer, loss_fn, num_updates, verbose=False):
        self.model.train()
        iter_loader = iter(data_loader)
        total_loss = 0
        for _ in range(num_updates):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(data_loader)
                batch = next(iter_loader)
            if len(batch) > 2:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            # Handle dictionary inputs for transformers
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            # Use dictionary unpacking for model call
            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_updates
        if verbose:
            print(f"Average loss over {num_updates} updates: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, data_loader, metric_fn, verbose=False):
        self.model.eval()
        total_metric = 0
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) > 2:
                    inputs, targets, _ = batch  # Ignore rest
                else:
                    inputs, targets = batch
                # Handle dictionary inputs for transformers
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Use dictionary unpacking for model call
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                total_metric += metric_fn(outputs, targets).item()
        avg_metric = total_metric / len(data_loader)
        if verbose:
            print(f"Evaluation metric: {avg_metric:.4f}")
        return avg_metric
    
    def evaluate_outputs(self, data_loader, metric_fn, verbose=False):
        self.model.eval()
        all_outputs = []
        all_targets = []
        total_metric = 0
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) > 2:
                    inputs, targets, _ = batch  # Ignore rest
                else:
                    inputs, targets = batch
                # Handle dictionary inputs for transformers
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Use dictionary unpacking for model call
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                total_metric += metric_fn(outputs, targets).item()
        avg_metric = total_metric / len(data_loader)
        if verbose:
            print(f"Evaluation metric: {avg_metric:.4f}")
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        return avg_metric, (all_outputs, all_targets)
    
    def get_params(self):
        return self.model.state_dict()
    
    def set_params(self, params):
        self.model.load_state_dict(params)
        
    def randomize_weights(self, init_type='uniform', a=-0.1, b=0.1, mean=0.0, std=0.1, verbose=False):
        """
        Randomize the weights and biases of the model.
        
        Args:
            init_type (str): Type of initialization ('uniform' or 'normal'). Default: 'uniform'.
            a (float): Lower bound for uniform initialization. Default: -0.1.
            b (float): Upper bound for uniform initialization. Default: 0.1.
            mean (float): Mean for normal initialization. Default: 0.0.
            std (float): Standard deviation for normal initialization. Default: 0.1.
            verbose (bool): If True, print confirmation of randomization. Default: False.
        
        Returns:
            None
        """
        def _randomize(m):
            if hasattr(m, 'weight'):
                if init_type == 'uniform':
                    nn.init.uniform_(m.weight, a=a, b=b)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, mean=mean, std=std)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}. Use 'uniform' or 'normal'.")
            if hasattr(m, 'bias') and m.bias is not None:
                if init_type == 'uniform':
                    nn.init.uniform_(m.bias, a=a, b=b)
                elif init_type == 'normal':
                    nn.init.normal_(m.bias, mean=mean, std=std)
        
        self.model.apply(_randomize)
        if verbose:
            print(f"Model weights randomized using {init_type} initialization.")
    
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
