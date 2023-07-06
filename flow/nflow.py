import torch
from torch.distributions.distribution import Distribution
import numpy as np
from abc import ABC, abstractmethod
from flow.layers import Layer


class Flow():
    def __init__(self, basedist: Distribution, flow_layers: list[Layer]):
        self.basedist = basedist
        self.flow_layers = flow_layers

    @abstractmethod
    def forward(self, z: torch.FloatTensor):
        pass

    @abstractmethod
    def inverse(self, x: torch.FloatTensor):
        pass

    @abstractmethod
    def rsample(self, shape: torch.Size):
        pass

    @abstractmethod
    def prob(self, x: torch.Tensor):
        pass
