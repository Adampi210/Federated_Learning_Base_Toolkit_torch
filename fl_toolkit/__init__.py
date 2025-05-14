# fl_toolkit/__init__.py

# Import from data_operations
from .data_operations import (
    BaseDataHandler,
    MNISTDataHandler,
    FashionMNISTDataHandler,
    CIFAR10DataHandler,
    CIFAR100DataHandler,
    PACSDataHandler,
    DigitsDGDataHandler,
    OfficeHomeDataHandler,
    DomainDrift,
)

# Import from federated
from .federated import (
    FederatedClient,
    FederatedDriftClient,
    DriftAgent,
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
    'DigitsDGDataHandler',
    'OfficeHomeDataHandler',
    
    # Drift operations
    'DomainDrift',
    
    # Federated
    'FederatedClient',
    'FederatedDriftClient',
    'DriftAgent',
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