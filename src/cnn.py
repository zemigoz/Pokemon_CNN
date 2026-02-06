import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 10),
        # ) # performed decently with a 87.29% val_acc, tiny val_loss, and every other metric at .87

        #conv2d(
        # in_channels = RGB is 3 but MNIST is grayscale hence 1
        # out_channels = number of patterns to learn. ur choice
        # kernel_size = the filter size in kernel_size x kernel_size tensor.shape (can also use a tuple for more control)
        # stride = step size or how many pixels you move. default is 1 but more downsamples/lowers output tensor.shape
        # padding = increases dim of image to affect dim of output. padding = 0 reduces while padding = 1 maintains dim. typically new cols/rows are 0s
        #   but other methods exist and can be specified
        # bias = uses bias terms boolean. false for less computation if followed by batch norm
        #) 

        # where i is the dimension of choice (x, y, z) or (length, height, width). does not affect batch_size nor channels
        # conv -> dim_i = math.floor([(dim_i_og + 2 * padding - dilation * (kernel_i - 1) - 1) / stride] + 1)
        # pool -> dim_i = math.floor([(dim_i_og + 2 * padding - kernel_i) / stride] + 1)

        self.conv_stack = nn.Sequential( # goes in as (batch_size, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False), # (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32), #reduces overfitting by normalizing out_channel outputs of convolutions only
            nn.ReLU(), #learn it now. rinse and repeat
            nn.MaxPool2d(2), #downsample using a kernel_size x kernel_size kernel with other args. (can use a tuple for more control)
            # maxpool reduced it to (batch_size, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), # [(14 + 2 * 2 - (5 - 1) - 1) / 1] + 1 = [(14 + 4 - (4) - 1) / 2] + 1 = 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (batch_size, 64, 7,7)

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), # [(14 + 2 * 2 - (5 - 1) - 1) / 1] + 1 = [(14 + 4 - (4) - 1) / 2] + 1 = 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2), # (batch_size, 64, 7,7)
            # nn.AdaptiveMaxPool2d((7,7)) #specify output. useful for varying image sizes and just not typing up conv2d()
        )

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256, bias=False), #since its flattened, it must use the product of (_, channels, **dim)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10) #finish with num_labels of choice
        )

        # self.fc_stack = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 7, 256),
        #     nn.ReLU(),
        #     nn.Linear(128, 10)
        # )


        # nn.Dropout(.5), #0.2-0.5 for hidden and 0.2 max for input and conv


    # Never call on this. Instead, call on the variable object NeuralNetwork() since it does .__call__() and it uses this method
    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)

        # pred_probab = nn.Softmax(dim=1)(logits)
        # y_pred = pred_probab.argmax(1)

        x = self.conv_stack(x)
        logits = self.fc_stack(x)

        return logits