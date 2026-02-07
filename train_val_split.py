import math
import time
import os
import gc
import unicodedata
import re
import shutil

from collections import Counter
from pathlib import Path
from PIL import Image

import torch
import torchvision
import wandb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor

from config import *

def sanitize(string: str) -> str:
    replacement_map = {
        'A©': 'e',
        'Æ': 'AE',
        'æ': 'ae',
        'ß': 'ss',
        "I´": ""
    }

    pokemon_name = unicodedata.normalize('NFD', string)

    pokemon_name = ''.join(c for c in pokemon_name if unicodedata.category(c) != 'Mn') #gets rid of accents
    pattern = re.compile("|".join(re.escape(k) for k in replacement_map.keys()))
    pokemon_name = pattern.sub(lambda m: replacement_map[m.group(0)], pokemon_name) #SO MUCH WORK JUST TO MAKE IT POKEMON AND NOT POKACMON, it aint an AC unit

    pokemon_name = re.sub(r"[^a-zA-Z0-9_-\u00E9]", "_", pokemon_name)
    pokemon_name = re.sub(r"_+", "_", pokemon_name)                  # collapse multiple underscores
    pokemon_name = pokemon_name.strip("_")

    return pokemon_name

def train_val_split(make_folders = True):
    dataset = datasets.ImageFolder(
        root=IMAGE_DIR,
        transform=None
    )

    indices = np.arange(len(dataset))

    # sanitize pokemon names in the csv
    pokemon_name_df = pd.read_csv(POKEMON_NAMES_CSV)
    allowed_names = set(
        pokemon_name_df["Name"]
        .astype(str)
        .apply(sanitize)
    ) 

    # sanitize dataset labels/classes just in case
    class_idx_to_name = {idx: sanitize(name) for idx, name in enumerate(dataset.classes)}

    # get dataset (paths, classes_idx) that are actual pokemon according to allowed_names
    filtered_samples = [
        (path, class_idx) 
        for path, class_idx in dataset.samples
        if class_idx_to_name[class_idx] in allowed_names
    ]

    # need at least 2 samples in each class
    class_counts = Counter(class_idx for _, class_idx in filtered_samples)
    valid_classes = {class_idx for class_idx, count in class_counts.items() if count >= 2}

    # filter again by getting rid rid of classes < 2
    filtered_samples = [
        (path, class_idx)
        for path, class_idx in filtered_samples
        if class_idx in valid_classes
    ]

    filtered_names = set(class_idx_to_name[idx] for idx in valid_classes) # get rid of dupes in case and get all pokemon names

    remaining_class_indices = sorted({class_idx for _, class_idx in filtered_samples}) # 
    old_idx_to_name = {idx: class_idx_to_name[idx] for idx in remaining_class_indices}

    new_class_names = sorted(old_idx_to_name.values())

    new_class_to_idx = {name: i for i, name in enumerate(new_class_names)}

    old_idx_to_new_idx = {old_idx: new_class_to_idx[name] for old_idx, name in old_idx_to_name.items()}

    dataset.classes = new_class_names
    dataset.class_to_idx = new_class_to_idx
    dataset.samples = [
        (path, old_idx_to_new_idx[class_idx])
        for path, class_idx in filtered_samples
    ]
    dataset.targets = [class_idx for _, class_idx in dataset.samples]

    print(f"Filtered dataset has {len(dataset.samples)} samples across {len(dataset.classes)}")

    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_TEST_SPLIT,
        stratify=dataset.targets,
        random_state=RNG_SEED
    )

    print(f"There are {len(train_idx)} training samples & {len(val_idx)} validation samples")

    if make_folders:
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

        all_samples = dataset.samples  # filtered dataset samples: list of (path, class_idx)
        class_idx_to_name = {idx: name for idx, name in enumerate(dataset.classes)}

        def copy_to_folder(samples, target_dir):
            for i, (path, class_idx) in enumerate(samples, start=1):
                class_name = class_idx_to_name[class_idx]
                class_folder = target_dir / class_name
                class_folder.mkdir(exist_ok=True, parents=True)
                # shutil.copy(path, class_folder)
                
                print(f"Opening path {path} with class index {class_idx}")
                img = Image.open(path)
                if img.mode == "P":
                    img = img.convert("RGBA")
                img = img.convert("RGB")

                save_path = class_folder / Path(path).name
                img.save(save_path)

        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        copy_to_folder(train_samples, TRAIN_DIR)
        copy_to_folder(val_samples, VALIDATION_DIR)

        print(f"Number of remaining classes with >=2 samples: {len(filtered_names)}")
        print(f"Number of remaining samples: {len(dataset.samples)}")

    else:
        val_dataset = Subset(dataset, val_idx)
        train_dataset = Subset(dataset, train_idx)
        return train_dataset, val_dataset
    
train_val_split(make_folders=True)
# if __name__ == '__main__':
#     start_time = time.time()
    
#     print(f'Time to run whole program: {time.time() - start_time}')