# Convolutional Neural Network Fun

Datasets can be found in the `torchvision` package.

# Table Of Contents
- [Convolutional Neural Network Fun](#convolutional-neural-network-fun)
- [Table Of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Running The Program](#running-the-program)
  - [Installations](#installations)
  - [Configuration](#configuration)
  - [Command To Run Program](#command-to-run-program)
  - [Runtime](#runtime)
- [Model \& Results](#model--results)
- [Discussion](#discussion)
- [Colaborators/Authors](#colaboratorsauthors)

# Introduction

This is a personal tutorial to PyTorch and Convolutional Neural Networks (CNNs). This project leans to a more informal written approach. The model, results, and code structure exceeded my expectations and encouraged me to work on a more formal and interesting project using CNNs. You can expect to see the next project soon.

# Running The Program

## Installations

Running the code first requires dependencies. Note <u>**the GPU-backend of PyTorch is not a part of the program's dependencies**</u>. Recent Fedora Linux updates and my wifi chipset make it mutually exclusive to support wifi or the GPU-backend. I chose the former. Please look into PyTorch's [download](https://pytorch.org/get-started/locally/) page if you would like GPU support.

Downloading all dependencies requires `conda`. Once installed, run the following:

```
conda env create -f cnn_environment.yml -n cnn
```

## Configuration

The program uses `wandb` (Weights & Biases) to log all information. A `wandb` account, unless commented out, is necessary to run the code. Please create an [account](https://wandb.ai/). Make a *wandb_api_key.txt* file (or rename it the *.txt* file and configure the `WANDB_API_KEY_PATH` variable as needed). Navigate the `wandb` website and create a project while copying the API key. Drop the API key into the *.txt* file you just made and edit `WANDB_PROJECT_NAME` to match the project name. 

We highly encourage to look and edit the configuration variables at the top of *main_cnn.py* and *main_tutorial.py* as you see fit. Feel free to edit other code but understand it is dense. Lastly, *analysis.py* is deprecated.

## Command To Run Program

Make sure you are in the directory of this *README.txt* when you run the following code:

```
conda activate cnn
```

Activates the environment. To run a simple neural network on the MNIST dataset:

```
python main_tutorial.py
```

To run a (relatively) simple convolutional neural network on Fashion MNIST:

```
python main_cnn.py
```

## Runtime

A long time. Note this <u>**only runs on the CPU**</u>. On a Dell Precision laptop, it took up to 45 minutes on *main_cnn.py* with the given hyperparameters/configuration variables. This is naive and highly encouraged to download GPU-backend PyTorch.

# Model & Results

The results are only for *main_cnn.py*.

| Hyperparameter| Value |
| ------------- | ----- |
| epoch         | 10    |
| batch_size    | 64    |
| learning_rate | 0.001 |
| rng_seed      | 314   |
| folds         | 5     |

With the convolutional neural network structure as:

```
ConvolutionalNeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (conv_stack): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (fc_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=3136, out_features=256, bias=False)
    (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=256, out_features=128, bias=False)
    (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.4, inplace=False)
    (9): Linear(in_features=128, out_features=10, bias=True)
  )
)
```


| Class       | Precision | Recall | F1     |
| ----------- | --------- | ------ | ------ |
| Ankle boot  | 0.9467    | 0.9770 | 0.9616 |
| Bag         | 0.9899    | 0.9850 | 0.9875 |
| Coat        | 0.8840    | 0.8380 | 0.8604 |
| Dress       | 0.9113    | 0.9250 | 0.9181 |
| Pullover    | 0.8342    | 0.9260 | 0.8777 |
| Sandal      | 0.9889    | 0.9770 | 0.9829 |
| Shirt       | 0.8262    | 0.7180 | 0.7683 |
| Sneaker     | 0.9654    | 0.9500 | 0.9577 |
| T-shirt/top | 0.8499    | 0.9060 | 0.8771 |
| Trouser     | 0.9889    | 0.9820 | 0.9854 |



| Metric                 | Value   |
| ---------------------- | ------- |
| Accuracy               | 0.9184  |
| Loss                   | 0.00345 |
| Macro Avg Precision    | 0.9186  |
| Macro Avg Recall       | 0.9184  |
| Macro Avg F1           | 0.9177  |
| Weighted Avg Precision | 0.9186  |
| Weighted Avg Recall    | 0.9184  |
| Weighted Avg F1        | 0.9177  |

# Discussion

Incredible! The model falls short on 4 ("Shirt", "Coat", "T-shirt/top", "Pullover") classes and quite poorly on "Shirt" particularly. While further analysis can be down on why this is happening, time and other interests may us move on from this project. We can only assume the very high complexity of these four classes is not within the current model's capabilities. 

# Colaborators/Authors

Authors
- Jeffrey Gomez