import torch
import torch.nn as nn
import torch.functional as F


def swish(x):
    return x*torch.sigmoid(x)


class DDPGmodel(nn.module):
    """Model designed for DDPG"""
    def __init__(self):
