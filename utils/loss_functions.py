import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(loss_name='cross_entropy', **kwargs):
    """
    Get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'cross_entropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_name == 'l1':
        return nn.L1Loss(**kwargs)
    elif loss_name == 'smooth_l1':
        return nn.SmoothL1Loss(**kwargs)
    elif loss_name == 'nll':
        return nn.NLLLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def calculate_accuracy(outputs, targets):
    """
    Calculate accuracy for classification tasks
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
    
    Returns:
        Accuracy percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total


def calculate_precision_recall_f1(outputs, targets, num_classes=10):
    """
    Calculate precision, recall and F1-score
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        precision, recall, f1 (macro averaged)
    """
    _, predicted = torch.max(outputs, 1)
    
    # Initialize counters
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    
    for i in range(num_classes):
        tp[i] = ((predicted == i) & (targets == i)).sum().item()
        fp[i] = ((predicted == i) & (targets != i)).sum().item()
        fn[i] = ((predicted != i) & (targets == i)).sum().item()
    
    # Calculate metrics (avoid division by zero)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Macro average
    precision_avg = precision.mean().item()
    recall_avg = recall.mean().item()
    f1_avg = f1.mean().item()
    
    return precision_avg, recall_avg, f1_avg


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization
    """
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_advanced_criterion(loss_name='cross_entropy', **kwargs):
    """
    Get advanced loss functions including custom ones
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'label_smoothing':
        return LabelSmoothingLoss(classes=kwargs.get('classes', 10), 
                                smoothing=kwargs.get('smoothing', 0.1))
    else:
        return get_criterion(loss_name, **kwargs)