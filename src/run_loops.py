from typing import Callable, Dict, cast
# from typing import Dict, Any

import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, Tensor
from torch.optim import Optimizer

from .tutorial_nn import NeuralNetwork

def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
):    
    device = next(model.parameters()).device

    total_loss = 0
    num_samples = 0

    all_preds = []
    all_labels = []

    # set model to training mode
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop (never will i code it again. ty DAG & PyTorch )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_samples += y.size(0)

        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def test_loop(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    labels_map
):
    # set model to evaluation mode (look into batch_normalization)
    model.eval()

    total_loss = 0
    num_samples = 0

    all_preds = []
    all_labels = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            total_loss += loss_fn(pred, y).item()
            num_samples += y.size(0)
    
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / num_samples
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # this is to simply have the type checker shut up about str | dict
    report = cast(
        Dict[str, Dict[str, float]],
        classification_report(
            y_true, 
            y_pred, 
            target_names=labels_map.values(), 
            output_dict=True
        )
    ) 

    return avg_loss, report