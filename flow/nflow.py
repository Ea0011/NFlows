from typing import Any
import torch
from torch.distributions.distribution import Distribution
import numpy as np
from abc import ABC, abstractmethod
from flow.layers import Layer, PlanarLayer, RadialLayer, RealNVPLayer, GlowLayer
import flow.nets as NN
from torch import nn


class Flow(nn.Module):
    def __init__(self, basedist: Distribution, flow_layers: nn.ModuleList):
        super(Flow, self).__init__()

        self.basedist = basedist
        self.flow_layers = flow_layers

    def forward(self, z: torch.FloatTensor):
        out = z
        log_det_jacobian = 0
        for layer in self.flow_layers:
            out = layer(out)
            log_det_jacobian += layer.log_det_jacobian

        return out, log_det_jacobian

    def inverse(self, x: torch.FloatTensor):
        out = x
        log_det_jacobian = 0
        for layer in reversed(self.flow_layers):
            out = layer.inverse(out)
            log_det_jacobian += layer.log_det_jacobian

        return out, log_det_jacobian

    def rsample(self, shape: torch.Size):
        z = self.basedist.rsample(shape)
        base_prob = self.basedist.log_prob(z)

        sample, log_det_jac = self.forward(z)
        prob = base_prob + log_det_jac.view_as(base_prob)

        return sample, prob

    def log_prob(self, x: torch.Tensor):
        z, log_det_jac = self.inverse(x)
        base_prob = self.basedist.log_prob(z)
        prob = base_prob.view(-1) + log_det_jac.view(-1)

        return prob.view(-1)


class PlanarFlow(Flow):
    def __init__(self, basedist: Distribution, flow_layers: nn.ModuleList):
        super().__init__(basedist, flow_layers)

    def forward(self, z):
        return Flow.forward(self, z)

    def inverse(self, x: torch.FloatTensor):
        return Flow.inverse(self, x)

    def rsample(self, shape: torch.Size):
        return Flow.rsample(self, shape)

    def log_prob(self, x: torch.Tensor):
        return Flow.log_prob(self, x)


class RaidalFlow(Flow):
    def __init__(self, basedist: Distribution, flow_layers: nn.ModuleList):
        super().__init__(basedist, flow_layers)

    def forward(self, z):
        raise NotImplementedError("This flow has no forward path")

    def inverse(self, x: torch.FloatTensor):
        return Flow.inverse(self, x)

    def rsample(self, shape: torch.Size):
        raise NotImplementedError("This flow has no forward path. So, cannot directly sample from it")

    def log_prob(self, x: torch.Tensor):
        return Flow.log_prob(self, x)


class RealNVPFlow(Flow):
    def __init__(self, basedist: Distribution, flow_layers: nn.ModuleList):
        super().__init__(basedist, flow_layers)

    def forward(self, z):
        return Flow.forward(self, z)

    def inverse(self, x: torch.FloatTensor):
        return Flow.inverse(self, x)

    def rsample(self, shape: torch.Size):
        return Flow.rsample(self, shape)

    def log_prob(self, x: torch.Tensor):
        return Flow.log_prob(self, x)


class Glow(Flow):
    def __init__(self, basedist: Distribution, flow_layers: nn.ModuleList):
        super().__init__(basedist, flow_layers)

    def forward(self, z):
        return Flow.forward(self, z)

    def inverse(self, x: torch.FloatTensor):
        return Flow.inverse(self, x)

    def rsample(self, shape: torch.Size):
        return Flow.rsample(self, shape)

    def log_prob(self, x: torch.Tensor):
        return Flow.log_prob(self, x)


class PlanarFlowBuilder:
    def __init__(self, dim=2, nlayers=4, basedist=torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))):
        layers = nn.ModuleList([PlanarLayer(dim, act="leaky_relu") for i in range(nlayers)])
        self.flow = PlanarFlow(basedist=basedist, flow_layers=layers)

    def __call__(self):
        return self.flow


class RadialFlowBuilder:
    def __init__(self, dim=2, nlayers=4, basedist=torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))):
        layers = nn.ModuleList([RadialLayer(dim) for i in range(nlayers)])
        self.flow = RaidalFlow(basedist=basedist, flow_layers=layers)

    def __call__(self):
        return self.flow


class RealNVPFlowBuilder:
    def __init__(self, masks, dim=2, hdim=32, basedist=torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)), net_class=NN.MLP):
        layers = nn.ModuleList([RealNVPLayer(masks[i], net_t=net_class(indim=dim, hdim=hdim), net_s=net_class(indim=dim, hdim=hdim)) for i in range(len(masks))])
        self.flow = RealNVPFlow(basedist=basedist, flow_layers=layers)

    def __call__(self):
        return self.flow


class GlowBuilder:
    def __init__(self, dim=2, hdim=32, nlayers=4, basedist=torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)), net_class=NN.MLP):
        layers = nn.ModuleList([GlowLayer(net_t=net_class(indim=dim // 2, hdim=hdim, squash=False),
                                          net_s=net_class(indim=dim // 2, hdim=hdim, squash=True), dim=dim) for i in range(nlayers)])
        self.flow = Glow(basedist=basedist, flow_layers=layers)

    def __call__(self):
        return self.flow


if __name__ == "__main__":
    f = PlanarFlowBuilder().flow
    f(torch.randn(4, 2))
