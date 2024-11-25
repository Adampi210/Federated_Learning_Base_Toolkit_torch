from fl_toolkit import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class SmallMLP(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)

class MediumCNN(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First block
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        
        # Second block
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, 2)
        
        # Third block
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        x = nn.functional.max_pool2d(x, 3)
        
        # Fully connected
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

class MNISTResNet(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        
    def forward(self, x):
        x = self.resnet(x)
        return nn.functional.log_softmax(x, dim=1)

N_SERVERS = 1
N_CLIENTS = 3
EPOCHS = 10
LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 256
LEARNING_RATE = 0.01

if __name__ == "__main__":
    # Model configurations to test
    model_configs = [
        ("Small MLP", SmallMLP()),
        ("Medium CNN", MediumCNN()),
        ("ResNet", MNISTResNet())
    ]
    
    # Load data once
    data = MNISTDataHandler()
    data.load_data('~/data/MNIST/')
    
    # Test each model architecture
    for model_name, model_arch in model_configs:
        
        print(f"\nTesting {model_name}")
        print("=" * 50)
        
        train_datasets, test_datasets = data.split_data(N_CLIENTS, strategy='iid')
        
        server = FederatedServer(model_arch)
        
        clients = []
        for i in range(N_CLIENTS):
            client = FederatedClient(
                client_id=i,
                model_architecture=type(model_arch)()
            )
            
            train_loader, test_loader = data.get_client_dataloaders(
                train_datasets[i], test_datasets[i], LOCAL_BATCH_SIZE
            )
            client.set_data(train_loader, test_loader)
            clients.append(client)
            server.add_client(client)
        
        print(f"Starting federated training for {model_name}...")
        for round in range(EPOCHS):
            print(f"\nRound {round+1}/{EPOCHS}")
            
            for client in clients:
                optimizer = optim.SGD(client.model.model.parameters(),
                                    lr=LEARNING_RATE,
                                    momentum=0.9)
                loss_fn = nn.CrossEntropyLoss()
                
                client.train(
                    epochs=LOCAL_EPOCHS,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    verbose=True
                )
            
            server.aggregate_models_FedAVG()
            server.distribute_model()
            
            print(f"\n{model_name} Client Evaluations:")
            for client in clients:
                acc = client.evaluate(metric_fn=accuracy_fn, verbose=False)
                print(f"Client {client.get_client_id()}: Accuracy = {acc:.4f}")
        
        # Final evaluation
        print(f"\nFinal Test Set Evaluation for {model_name}:")
        test_loader = data.get_test_loader(batch_size=LOCAL_BATCH_SIZE)
        for client in clients:
            acc = client.model.evaluate(test_loader, accuracy_fn, verbose=False)
            print(f"Client {client.get_client_id()}: Test Accuracy = {acc:.4f}")