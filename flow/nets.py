import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class ResNet(nn.Module):  # this layer also has batch norm for stability
    def __init__(self, indim, hdim, squash=True):
        super(ResNet, self).__init__()

        self.indim = indim
        self.hdim = hdim
        self.squash = squash

        self.fc1 = nn.Linear(indim, hdim)
        self.bn1 = nn.LayerNorm(hdim, elementwise_affine=False)

        self.fc2 = nn.Linear(hdim, hdim)
        self.bn2 = nn.LayerNorm(hdim, elementwise_affine=False)

        self.fc3 = nn.Linear(hdim, hdim)
        self.bn3 = nn.LayerNorm(hdim, elementwise_affine=False)

        self.fc4 = nn.Linear(hdim, indim)

    def forward(self, x):
        x_ = F.leaky_relu(self.bn1(self.fc1(x)))
        x_ = F.leaky_relu(self.bn2(self.fc2(x_)) + x_)
        x_ = F.leaky_relu(self.bn3(self.fc3(x_)))
        x_ = self.fc4(x_)
        x_ = torch.tanh(x_) if self.squash else x_

        return x_


class MLP(nn.Module):
    def __init__(self, indim, hdim, squash=True):
        super(MLP, self).__init__()

        self.squash = squash

        self.fc1 = nn.Linear(indim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc4 = nn.Linear(hdim, indim)

    def forward(self, x):
        x_ = F.leaky_relu(self.fc1(x))
        x_ = F.leaky_relu(self.fc2(x_))
        x_ = F.leaky_relu(self.fc3(x_))
        x_ = self.fc4(x_)
        x_ = torch.tanh(x_) if self.squash else x_

        return x_
