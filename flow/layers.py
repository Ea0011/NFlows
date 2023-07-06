import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        pass

    def forward(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not inverse flow")


class RadialFlow(Layer):
    def __init__(self, dim=2):
        super(RadialFlow, self).__init__()
        self.dim = dim

        self.z0 = nn.Parameter(torch.Tensor(self.dim,)) # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(torch.Tensor(1,)) # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(torch.Tensor(1,)) # Scaler used to indireclty parametrized \beta

        stdv = 1. / math.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.z0.data.uniform_(-stdv, stdv)

        self.log_det_jacobian = 0

    def forward(self, z: torch.Tensor):
        B, D = z.shape

        alpha = F.softplus(self.pre_alpha)
        beta = -alpha + F.softplus(self.pre_beta)

        radius = torch.norm(z - self.z0, p=2, dim=1).unsqueeze(1)
        offset = z - self.z0
        h = 1 / (self.pre_alpha + radius)

        out = z + (beta * h) * offset
        hp = - 1 / (alpha + radius) ** 2
        det_jac = (1 + beta*h)**(D - 1) * (1 + beta * h + beta * hp * radius)
        self.log_det_jacobian = torch.log(det_jac.abs() + 1e-8).view(-1)

        return out


class PlanarFlow(Layer):
    def __init__(self, dim, h=F.relu):
        super(PlanarFlow, self).__init__()
        self.h = h
        self.hp = lambda x: 1 - self.h(x) ** 2
        self.fn = nn.Linear(dim, dim)
        self.log_det_jacobian = 0
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.scale.data.uniform_(-0.01, 0.01)

    def forward(self, z: torch.Tensor):
        out = self.fn(z)
        detjac = torch.abs(1 + torch.mm(self.hp(out) @ self.fn.weight, self.scale.T))
        self.log_det_jacobian = torch.log(detjac + 1e-8).view(-1)

        return z + self.scale * self.h(out)
