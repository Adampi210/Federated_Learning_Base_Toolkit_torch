# fl_toolkit/__init__.py

# Import from data_operations
from .data_operations import (
    BaseDataHandler,
    MNISTDataHandler,
    FashionMNISTDataHandler,
    CIFAR10DataHandler,
    CIFAR100DataHandler,
    DataDrift,
    ConceptDrift, 
    DriftedDataset
)

# Import from federated
from .federated import (
    FederatedClient,
    FederatedDriftClient,
    FederatedServer
)

# Import from model operations
from .model_operations import (
    RandomProjectionCompressor, 
    SVDLiteCompressor, 
    TensorTrainCompressor
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
    
    # Drift operations
    'DataDrift',
    'ConceptDrift', 
    'DriftedDataset',
    
    # Federated
    'FederatedClient',
    'FederatedDriftClient',
    'FederatedServer',
    
    # Models
    'BaseModelArchitecture',
    'BaseNeuralNetwork',
    
    # Model Operations
    'RandomProjectionCompressor',
    'SVDLiteCompressor',
    'TensorTrainCompressor',
    
    # Utils
    'accuracy_fn',
    'precision_fn',
    'recall_fn',
    'f1_score_fn',
]