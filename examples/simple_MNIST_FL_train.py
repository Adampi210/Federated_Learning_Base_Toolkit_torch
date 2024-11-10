from fl_toolkit import *
import torch
import torch.nn as nn
import torch.optim as optim

N_SERVERS = 1
N_CLIENTS = 5
EPOCHS = 5
LOCAL_EPOCHS = 2
LOCAL_BATCH_SIZE = 32
LEARNING_RATE = 0.01

class MNISTNet(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Changed from Dropout2d to regular Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

if __name__ == "__main__":
    # Load and split data
    data = MNISTDataHandler()
    data.load_data('~/data/MNIST/')
    train_datasets, test_datasets = data.split_data(N_CLIENTS, strategy='iid')
    
    # Create server
    server = FederatedServer(MNISTNet())
    
    # Create clients
    clients = []
    for i in range(N_CLIENTS):
        # Initialize client
        client = FederatedClient(
            client_id=i,
            model_architecture=MNISTNet()
        )
        
        # Set client's data
        # First, get the dataloaders for training and testing data
        train_loader, test_loader = data.get_client_dataloaders(
            train_datasets[i], 
            test_datasets[i],
            LOCAL_BATCH_SIZE
        )
        # Then set the data
        client.set_data(train_loader, test_loader)
        clients.append(client)
        server.add_client(client)
    
    # Training loop
    print("Starting federated training...")
    for round in range(EPOCHS):
        print(f"\nRound {round+1}/{EPOCHS}")
        
        # Train each client
        for client in clients:
            optimizer = optim.SGD(client.model.model.parameters(), 
                                lr=LEARNING_RATE, 
                                momentum=0.9)
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
        
        # Evaluate each client
        print("\nClient Evaluations:")
        for client in clients:
            acc = client.evaluate(metric_fn=accuracy_fn, verbose=False)
            print(f"Client {client.get_client_id()}: Accuracy = {acc:.4f}")
    
    # Final evaluation on test set
    print("\nFinal Test Set Evaluation:")
    test_loader = data.get_test_loader(batch_size=LOCAL_BATCH_SIZE)
    for client in clients:
        acc = client.model.evaluate(test_loader, accuracy_fn, verbose=False)
        print(f"Client {client.get_client_id()}: Test Accuracy = {acc:.4f}")
    
    print("\nTraining completed!")