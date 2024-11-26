# utils/metrics.py
import torch
from fl_toolkit import *

# Calculate accuracy for prediction
def accuracy_fn(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()

# Calculate the precision for multi-class classification
def precision_fn(outputs, targets):
    predictions = outputs.argmax(dim=1)
    num_classes = outputs.shape[1]
    device = outputs.device
   
    precisions = []
    for class_id in range(num_classes):
        true_positives = ((predictions == class_id) & (targets == class_id)).float().sum()
        predicted_positives = (predictions == class_id).float().sum()
       
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
    
    if precisions:
        return sum(precisions).div(len(precisions))
    return torch.zeros(1, device=device)

# Calculate recall for multi-class classification
def recall_fn(outputs, targets):
    predictions = outputs.argmax(dim=1)
    num_classes = outputs.shape[1]
    device = outputs.device
   
    recalls = []
    for class_id in range(num_classes):
        true_positives = ((predictions == class_id) & (targets == class_id)).float().sum()
        actual_positives = (targets == class_id).float().sum()
       
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
    
    if recalls:
        return sum(recalls).div(len(recalls))
    return torch.zeros(1, device=device)

# Calculate F1 score
def f1_score_fn(outputs, targets):
    prec = precision_fn(targets, outputs)
    rec = recall_fn(targets, outputs)
    device = outputs.device
   
    if prec + rec > 0:
        return (2 * prec * rec).div(prec + rec)
    return torch.zeros(1, device=device)

