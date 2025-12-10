# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    # TODO: Implement a simple linear model
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    flatlayer = nn.Flatten()
    C,H,W = input_size
    linearlayer = nn.Linear(C*H*W,num_classes)

    layers = [flatlayer, linearlayer]
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    return nn.Sequential(*layers)


def FFN(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    num_layers = cfg.get("num_layers", 1)
    num_hidden = cfg.get("num_hidden", 128)
    use_dropout = cfg.get("use_dropout", False)
    # TODO: Implement a simple linear model
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    
    C,H,W = input_size
    flatlayer = nn.Flatten()
    linearlayer = nn.Linear(C*H*W,num_hidden)
    dropout = nn.Dropout(p=0.1)
    activationlayer = nn.ReLU()
    hiddenlayer1 = nn.Linear(num_hidden,num_hidden)
    hiddenlayer2 = nn.Linear(num_hidden,num_hidden)
    outlayer = nn.Linear(num_hidden,num_classes)
    layers = [flatlayer, linearlayer, activationlayer, dropout, hiddenlayer1, activationlayer, dropout, hiddenlayer2, activationlayer, dropout, outlayer]
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return nn.Sequential(*layers)
