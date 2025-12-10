# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]




class FancyCNN(nn.Module):
    """
    A fancy CNN model with :
        - stacked 3x3 convolutions
        - convolutive down sampling
        - a global average pooling at the end
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        layers = []
        cin = input_size[0]
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # TODO: Implement the model
        N=7
        for i in range(N):
            temp_cin = cin * 2**(i)
            layers = layers + conv_relu_bn(temp_cin,temp_cin)
            layers = layers + conv_relu_bn(temp_cin,2*temp_cin)
            layers = layers + conv_down(2*temp_cin,2*temp_cin)
        self.model = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.outlinear = nn.Linear(temp_cin*2, num_classes)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def forward(self, x):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # TODO: Implement the forward pass
        x = self.model(x)
        x = self.gap(x).flatten(1)
        x = self.outlinear(x)

        return x
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
