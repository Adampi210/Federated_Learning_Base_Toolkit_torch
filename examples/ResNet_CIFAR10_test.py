from fl_toolkit import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

N_SERVERS = 1
N_CLIENTS = 3
EPOCHS = 20
LOCAL_EPOCHS = 2
LOCAL_BATCH_SIZE = 256
LEARNING_RATE = 0.01

class ResNet18Classifier(BaseModelArchitecture):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the last layer for CIFAR-10
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Load and split data
    data = CIFAR10DataHandler()
    data.load_data('~/data/CIFAR10/', download=True)
    train_datasets, test_datasets = data.split_data(N_CLIENTS, strategy='iid')
    
    # Create server
    server = FederatedServer(ResNet18Classifier())
    
    # Create clients
    clients = []
    for i in range(N_CLIENTS):
        # Initialize client
        client = FederatedClient(
            client_id=i,
            model_architecture=ResNet18Classifier()
        )
        
        # Set client's data
        train_loader, test_loader = data.get_client_dataloaders(
            train_datasets[i],
            test_datasets[i],
            LOCAL_BATCH_SIZE
        )
        client.set_data(train_loader, test_loader)
        clients.append(client)
        server.add_client(client)
    
    # Training loop
    print("Starting federated training with ResNet18...")
    for round in range(EPOCHS):
        print(f"\nRound {round+1}/{EPOCHS}")
        
        # Train each client
        for client in clients:
            # Using SGD with momentum and weight decay for ResNet
            optimizer = optim.Adam(
                client.model.model.parameters(),
                lr=LEARNING_RATE,
            )
            loss_fn = nn.CrossEntropyLoss()
            
            # Local training
            client.train(
                epochs=LOCAL_EPOCHS,
                optimizer=optimizer,
                loss_fn=loss_fn,
                verbose=True
            )
        
        # Server aggregates models
        server.aggregate_models_FedAVG()
        
        # Distribute updated model to clients
        server.distribute_model()
        
        # Evaluate each client on their test set
        print("\nClient Evaluations:")
        for client in clients:
            acc = client.evaluate(metric_fn=accuracy_fn, verbose=False)
            print(f"Client {client.get_client_id()}: Test Accuracy = {acc:.4f}")
    
    # Final evaluation using each client's test set
    print("\nFinal Test Set Evaluation:")
    for client in clients:
        acc = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        print(f"Client {client.get_client_id()}: Final Test Accuracy = {acc:.4f}")
    
    print("\nTraining completed!")