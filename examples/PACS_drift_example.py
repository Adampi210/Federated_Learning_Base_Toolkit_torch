import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from fl_toolkit import *

class PACSCNN(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 7)  # 7 classes in PACS
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_dataset():
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

def test_single_client():
    # Initialize dataset
    pacs_handler = PACSDataHandler()
    pacs_handler.load_data()
    
    # Get data for initial domain (photo)
    train_data, test_data = pacs_handler.train_dataset, pacs_handler.test_dataset

    # Create separate drift configurations for train and test
    train_drift = PACSDomainDrift(
        source_domains=['photo', 'cartoon'],
        target_domains=['sketch', 'art_painting'],
        drift_rate=0.2,
        desired_size=len(train_data) / 2
    )
    
    test_drift = PACSDomainDrift(
        source_domains=['photo', 'cartoon'],
        target_domains=['sketch', 'art_painting'],
        drift_rate=0.2,
        desired_size=len(test_data) / 2
    )
    
    # Create client with separate drift handlers
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=PACSCNN,  # Pass class, not instance
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )
    
    # Set data
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    client.set_data(train_loader, test_loader)

    # Access the actual model parameters
    optimizer = optim.Adam(client.model.model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Initial training on source domain...")
    for epoch in range(5):
        client.train(epochs=1, optimizer=optimizer, loss_fn=criterion)
        acc = client.evaluate(
            metric_fn=lambda outputs, targets: (outputs.argmax(dim=1) == targets).float().mean()
        )
        print(f"Epoch {epoch}, Accuracy: {acc:.4f}")

    print("\nApplying drift...")
    for step in range(5):
        client.apply_train_drift()  # Apply drift to training data
        client.apply_test_drift()   # Apply drift to testing data
        acc = client.evaluate(
            metric_fn=lambda outputs, targets: (outputs.argmax(dim=1) == targets).float().mean()
        )
        print(f"Drift step {step}, Accuracy: {acc:.4f}\n\n")
    
if __name__ == "__main__":
    # First test the dataset
    print("Testing the dataset")
    # test_dataset()
    
    print("Testing single client scenario:")
    test_single_client()
    
