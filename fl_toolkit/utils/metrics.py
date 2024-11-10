# utils/metrics.py

import torch
from fl_toolkit import *

# Calculate accuracy for prediction
def accuracy_fn(targets, outputs):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()

# Calculate the precision for milti-class classification
def precision_fn(targets, outputs):
    predictions = outputs.argmax(dim=1)
    num_classes = outputs.shape[1]
    
    precisions = []
    for class_id in range(num_classes):
        true_positives = ((predictions == class_id) & (targets == class_id)).float().sum()
        predicted_positives = (predictions == class_id).float().sum()
        
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
            
    return torch.tensor(sum(precisions) / len(precisions) if precisions else 0.0)

# Calculate recall for multi-class classification
def recall_fn(targets, outputs):
    predictions = outputs.argmax(dim=1)
    num_classes = outputs.shape[1]
    
    recalls = []
    for class_id in range(num_classes):
        true_positives = ((predictions == class_id) & (targets == class_id)).float().sum()
        actual_positives = (targets == class_id).float().sum()
        
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
            
    return torch.tensor(sum(recalls) / len(recalls) if recalls else 0.0)

# Calculate F1 score
def f1_score_fn(targets, outputs):
    prec = precision_fn(targets, outputs)
    rec = recall_fn(targets, outputs)
    
    return torch.tensor(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0)

# Calculate cross-entropy loss
def loss_fn(targets, outputs):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, targets)
