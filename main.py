import math
import time
import os
import gc

from collections import Counter
from pathlib import Path

import torch
import torchvision
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor

from src.cnn import ConvolutionalNeuralNetwork
from src.run_loops import *
from config import *

# --------------------------------------------------------------------------- #
# MAIN WALK
# --------------------------------------------------------------------------- #
torch.manual_seed(RNG_SEED)

def main():
    if torch.cuda.is_available():
        print("CUDA device found")
        device_string = "cuda"
    else:
        print("No CUDA found. Resorting to CPU")
        device_string = "cpu"
    device = torch.device(device_string)

    if USE_F16_SCALER:
        scaler = torch.amp.GradScaler(device=device)
    else:
        scaler = None

    os.environ["WANDB_API_KEY"] = WANDB_API_KEY_PATH.read_text().strip()
    wandb_group_name = Path(f"MNIST-{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}")
    

    train_transform = transforms.Compose([
        # transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # z-score
    ])

    test_transform = transforms.Compose([
        # transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading Training Data")
    training_data = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    print("Loading Validation Data")
    validate_data = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=test_transform
    )

    labels_map = training_data.class_to_idx
    labels_map = {idx: cls for cls, idx in labels_map.items()} 

    # labels_names = [labels_map[i] for i in training_data.targets]
    labels = training_data.targets

    # class_distribution(training_data=training_data)

    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    print("Processing Training data to DataLoaders")
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print("Processing Validation data to DataLoaders")
    validate_dataloader = DataLoader(validate_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    train_dataset = train_dataloader.dataset


    # labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    ####### KFOLD STRATIFY
    print("Running KFolds now")
    skf = StratifiedKFold(
        n_splits=FOLDS, 
        shuffle=True, 
        random_state=RNG_SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, labels), start=1):
        print(f"Fold {fold}/{FOLDS}")


        # initiate all model related things
        model = ConvolutionalNeuralNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
        # scheduler = # may wish to add. changes learning rate according to whatever algo/formula u choose like cosine, exp, or small loss steps
        
        # wandb initialize
        wandb.init(
            project=WANDB_PROJECT_NAME, 
            group=str(wandb_group_name),
            name=f"fold_{fold}", 
            config={
                "epoch": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": ALPHA,
                "num_folds": FOLDS,
                "model_structure": str(model)
            },
            reinit="finish_previous"
        )

        # split into train & test
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_kfold_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_kfold_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # epoch training runs
        # epoch_train_losses = []
        # epoch_train_accuracies = []
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_loop(
                dataloader=train_kfold_loader, 
                model=model, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                scaler=scaler
            )
            wandb.log({
                # "fold": fold,
                # "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc
            })

            # epoch_train_losses.append(train_loss)
            # epoch_train_accuracies.append(train_acc)


        test_loss, metrics_report = test_loop(
            dataloader=test_kfold_loader, 
            model=model, 
            loss_fn=loss_fn, 
            labels_map=labels_map,
            scaler=scaler
        )

        macro_avg = metrics_report["macro avg"]
        weighted_avg = metrics_report["weighted avg"]

        wandb.log({
            # "fold": fold,

            "kfold/test_loss": test_loss,
            "kfold/test_accuracy": metrics_report["accuracy"],

            "kfold/macro_avg/precision": macro_avg["precision"],
            "kfold/macro_avg/recall": macro_avg["recall"],
            "kfold/macro_avg/f1": macro_avg["f1-score"],

            "kfold/weighted_avg/precision": weighted_avg["precision"],
            "kfold/weighted_avg/recall": weighted_avg["recall"],
            "kfold/weighted_avg/f1": weighted_avg["f1-score"],
        })

        # for class_name, class_metrics in metrics_report.items():
        #     if class_name not in ["macro avg", "weighted avg", "accuracy"]:
        #         wandb.log({
        #             f"kfold/{class_name}_precision": class_metrics["precision"],
        #             f"kfold/{class_name}_recall": class_metrics["recall"],
        #             f"kfold/{class_name}_f1": class_metrics["f1-score"]
        #         })

        wandb.finish()
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


    ##### VALIDATATION
    model = ConvolutionalNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    # epoch training runs
    epoch_train_losses = []
    epoch_train_accuracies = []
    for _ in range(EPOCHS):
        train_loss, train_acc = train_loop(
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn, 
            optimizer=optimizer
        )
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)

    # test on kfold test set (not validation set)
    val_loss, metrics_report = test_loop(
        dataloader=validate_dataloader, 
        model=model, 
        loss_fn=loss_fn, 
        labels_map=labels_map,
        scaler=scaler
    )

    macro_avg = metrics_report["macro avg"]
    weighted_avg = metrics_report["weighted avg"]

    wandb.init(
        project=WANDB_PROJECT_NAME, 
        group=str(wandb_group_name),
        name=f"validation_set", 
        config={
            "epoch": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": ALPHA,
            "model_structure": str(model)
            # "num_folds": FOLDS,
        },
        reinit="finish_previous"
    )

    wandb.log({
        "val/loss": val_loss,
        "val/accuracy": metrics_report["accuracy"],

        "val/macro_avg/precision": macro_avg["precision"],
        "val/macro_avg/recall": macro_avg["recall"],
        "val/macro_avg/f1": macro_avg["f1-score"],

        "val/weighted_avg/precision": weighted_avg["precision"],
        "val/weighted_avg/recall": weighted_avg["recall"],
        "val/weighted_avg/f1": weighted_avg["f1-score"],
    })

    # for class_name, class_metrics in metrics_report.items():
    #     if class_name not in ["macro avg", "weighted avg", "accuracy"]:
    #         wandb.log({
    #             f"val/{class_name}_precision": class_metrics["precision"],
    #             f"val/{class_name}_recall": class_metrics["recall"],
    #             f"val/{class_name}_f1": class_metrics["f1-score"]
    #         })

    wandb.finish()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time to run whole program: {time.time() - start_time}')