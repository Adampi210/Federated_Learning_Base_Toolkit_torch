# fl_toolkit/__init__.py

# Import from data_operations
from .data_operations import (
    BaseDataHandler,
    MNISTDataHandler,
    FashionMNISTDataHandler,
    CIFAR10DataHandler,
    CIFAR100DataHandler,
    PACSDataHandler,
    PACSDomainDrift,
    CIFAR10DomainDrift,
    CIFAR10DriftTypes
)

# Import from federated
from .federated import (
    FederatedClient,
    FederatedDriftClient,
    FederatedDriftClientCIFAR10,
    FederatedCompressedClient,
    FederatedServer,
    FederatedCompressedServer
)

# Import from model operations
from .model_operations import (
    WeightPruner,
    SplitModelCompressor
)

# Import from models
from .models import (
    BaseModelArchitecture,
    BaseNeuralNetwork
)

# Import from utils
from .utils import (
    accuracy_fn,
    precision_fn,
    recall_fn,
    f1_score_fn,
)

__all__ = [
    # Data operations
    'BaseDataHandler',
    'MNISTDataHandler',
    'FashionMNISTDataHandler',
    'CIFAR10DataHandler',
    'CIFAR100DataHandler',
    'PACSDataHandler',
    
    # Drift operations
    'PACSDomainDrift',
    'CIFAR10DomainDrift',
    'CIFAR10DriftTypes',
    
    # Federated
    'FederatedClient',
    'FederatedDriftClient',
    'FederatedDriftClientCIFAR10',
    'FederatedCompressedClient',
    'FederatedServer',
    'FederatedCompressedServer',
    
    # Models
    'BaseModelArchitecture',
    'BaseNeuralNetwork',
    
    # Model Operations
    'WeightPruner',
    'SplitModelCompressor',
    
    # Utils
    'accuracy_fn',
    'precision_fn',
    'recall_fn',
    'f1_score_fn',
]