import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import defaultdict
from fl_toolkit import *
    
def main():

    
    def visualize_sample(image, label, domain, categories):
        """Helper function to visualize a sample"""
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = image * std + mean
            img = img.permute(1, 2, 0).clip(0, 1)
        else:
            img = image
            
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Domain: {domain}\nClass: {categories[label]}")
        plt.axis('off')
        plt.savefig(f'ex_{label}_{domain}.png')

    def get_random_samples(handler, domain=None, label=None, n_samples=1):
        """Get random samples from the dataset with optional domain and label filtering"""
        if handler.train_dataset is None:
            raise ValueError("Dataset not loaded")
            
        # Get indices that match the criteria
        indices = range(len(handler.train_dataset))
        
        if domain is not None or label is not None:
            indices = [
                i for i in indices
                if (domain is None or handler.train_dataset.data[i][2] == domain) and
                (label is None or handler.train_dataset.data[i][1] == label)
            ]
        
        if not indices:
            raise ValueError(f"No samples found for domain={domain}, label={label}")
            
        # Randomly select samples
        selected_indices = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)
        return [handler.train_dataset[idx] for idx in selected_indices]

    # Initialize handler
    handler = PACSDataHandler()
    
    print("Loading PACS dataset...")
    handler.load_data()
    
    # Print dataset info
    info = handler.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test getting samples from each domain
    print("\nTesting random samples from each domain...")
    for domain in handler.domains:
        print(f"\nGetting sample from domain: {domain}")
        try:
            samples = get_random_samples(handler, domain=domain, n_samples=1)
            for image, label, domain in samples:
                visualize_sample(image, label, domain, handler.categories)
        except ValueError as e:
            print(f"Error: {e}")
    
    # Test getting samples from each class
    print("\nTesting random samples from each class...")
    for i, category in enumerate(handler.categories):
        print(f"\nGetting sample from class: {category}")
        try:
            samples = get_random_samples(handler, label=i, n_samples=1)
            for image, label, domain in samples:
                visualize_sample(image, label, domain, handler.categories)
        except ValueError as e:
            print(f"Error: {e}")
    
    # Test domain splitting
    print("\nTesting domain splitting...")
    for domain in handler.domains:
        train_domain, test_domain = handler.get_domain_data(domain)
        print(f"\nDomain: {domain}")
        print(f"Train samples: {len(train_domain)}")
        print(f"Test samples: {len(test_domain)}")

if __name__ == "__main__":
    main()