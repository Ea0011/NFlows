import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        pass

    def forward(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError(f"{self.__class__.__name__} has no inverse flow")


class RadialLayer(Layer):
    """Radial flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)
    Radial flow modifies density around a reference point z0

    ```
        f(z) = z + bh(a, r)(z - z0)
    ```
    """
    def __init__(self, shape=2):
        super(RadialLayer, self).__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer("d", self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / np.prod(shape)
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)
        self.z_0 = nn.Parameter(torch.randn(shape)[None])
        self.log_det_jacobian = 0

    def inverse(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.linalg.vector_norm(dz, dim=list(range(1, self.z_0.dim())), keepdim=True)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = -beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        self.log_det_jacobian = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        self.log_det_jacobian = self.log_det_jacobian.reshape(-1)
        return z_


class PlanarLayer(Layer):
    """Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)
    Planar flow shifts the input according to a non linearity and a linear transformation

    ```
        f(z) = z + u * h(w * z + b)
    ```
    """

    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """Constructor of the planar flow

        Args:
          shape: shape of the latent variable z
          h: nonlinear function h of the planar flow (see definition of f above)
          u,w,b: optional initialization for parameters
        """
        super(PlanarLayer, self).__init__()
        lim_w = np.sqrt(2.0 / np.prod(shape))
        lim_u = np.sqrt(2)
        self.log_det_jacobian = 0

        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))
            nn.init.uniform_(self.b, -0.01, 0.01)

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("Nonlinearity is not implemented.")

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim())),
                        keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)  # constraint w.T * u > -1
        if self.act == "tanh":
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + u * self.h(lin)
        self.log_det_jacobian = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.reshape(-1))))

        return z_

    def inverse(self, z):
        if self.act != "leaky_relu":
            raise NotImplementedError("This flow has no algebraic inverse.")
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        a = (lin < 0) * (
            self.h.negative_slope - 1.0
        ) + 1.0  # absorb leakyReLU slope into u
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)
        dims = [-1] + (u.dim() - 1) * [1]
        u = a.reshape(*dims) * u
        inner_ = torch.sum(self.w * u, list(range(1, self.w.dim())))
        z_ = z - u * (lin / (1 + inner_)).reshape(*dims)
        self.log_det_jacobian = -torch.log(torch.abs(1 + inner_))

        return z_


class RealNVPLayer(Layer):
    def __init__(self, mask, net_t, net_s):
        super(RealNVPLayer, self).__init__()
        self.net_t = net_t  # computes translation
        self.net_s = net_s  # computes scale
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.log_det_jacobian = 0

    def forward(self, z):
        # compute translation and scale based on the remaining part defined by the binary mask
        translation = self.net_t(z * self.mask)
        scale = self.net_s(z * self.mask)
        out = self.mask * z + (1. - self.mask) * (z * scale.exp() + translation)

        self.log_det_jacobian = torch.sum((1. - self.mask) * scale, -1)

        return out

    def inverse(self, x):
        scale = self.net_s(x * self.mask)
        translation = self.net_t(x * self.mask)
        z = x * self.mask + (1 - self.mask) * ((x - translation) * torch.exp(-scale))

        self.log_det_jacobian = torch.sum((1. - self.mask) * (-scale), -1)

        return z


# %--------%
# GLow layers

class ActNorm(Layer):
    def __init__(self, dim=2):
        super(ActNorm, self).__init__()

        self.t = nn.Parameter(torch.randn((1, dim)), requires_grad=True)
        self.s = nn.Parameter(torch.randn((1, dim)), requires_grad=True)
        self.init_done = False
        self.log_det_jacobian = 0

    def forward(self, z):
        if not self.init_done:
            s = -torch.log(z.std(dim=0, keepdim=True))
            t = (-z * torch.exp(s)).mean(dim=0, keepdim=True)

            self.t.data = t.detach()
            self.s.data = s.detach()

            self.init_done = True

        out = z * torch.exp(self.s) + self.t
        self.log_det_jacobian = torch.sum(self.s, dim=1)

        return out

    def inverse(self, x):
        if not self.init_done:
            s = -torch.log(x.std(dim=0, keepdim=True))
            t = (-x * torch.exp(s)).mean(dim=0, keepdim=True)

            self.t.data = t.detach()
            self.s.data = s.detach()

            self.init_done = True

        out = (x - self.t) * torch.exp(-self.s)
        self.log_det_jacobian = torch.sum(-self.s, dim=1)

        return out


class Invertable1x1Conv(Layer):
    def __init__(self, dim=2):
        super(Invertable1x1Conv, self).__init__()
        self.dim = dim

        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        LU, pivots = torch.linalg.lu_factor(Q)
        P, L, U = torch.lu_unpack(LU, pivots)
        self.P = P

        # Separate the scale component from the U matrix for logdet compuration
        S = U.diag()

        self.U = nn.Parameter(U.triu(diagonal=1))
        self.S = nn.Parameter(S)
        self.L = nn.Parameter(L)

        self.log_det_jacobian = 0

    def _assemble_W(self):
        # Make sure that L, U are proper triangular matrices durinng optimization
        U = torch.triu(self.U, diagonal=1)
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))

        return self.P @ (L @ (U + torch.diag(self.S)))

    def forward(self, z):
        W = self._assemble_W()
        out = z @ W

        self.log_det_jacobian = torch.sum(torch.log(torch.abs(self.S)))

        return out

    def inverse(self, x):
        W_inv = self._assemble_W().inverse()
        out = x @ W_inv

        self.log_det_jacobian = -torch.sum(torch.log(torch.abs(self.S)))

        return out


class AffineCouplingLayer(Layer):
    def __init__(self, net_s, net_t):
        super(AffineCouplingLayer, self).__init__()
        self.net_t = net_t  # computes translation
        self.net_s = net_s  # computes scale
        self.log_det_jacobian = 0

    def forward(self, z):
        z1, z2 = z[:, ::2].clone(), z[:, 1::2].clone()   # split in two parts
        t = self.net_t(z1)
        s = self.net_s(z1)

        x2 = z2 * s.exp() + t
        x1 = z1

        out = torch.cat([x1, x2], dim=1)

        self.log_det_jacobian = torch.sum(s, dim=1)

        return out

    def inverse(self, x):
        x1, x2 = x.clone()[:, ::2], x.clone()[:, 1::2]

        t = self.net_t(x1)
        s = self.net_s(x1)

        z2 = (x2 - t) * torch.exp(-s)
        z1 = x1

        out = torch.cat([z1, z2], dim=1)

        self.log_det_jacobian = torch.sum(-s, dim=1)

        return out


class GlowLayer(Layer):
    def __init__(self, net_t, net_s, dim=2, norm=False):
        super(GlowLayer, self).__init__()
        self.dim = dim
        self.net_t = net_t
        self.net_s = net_s
        self.norm = norm

        self.act_norm = ActNorm(dim)
        self.conv = Invertable1x1Conv(dim)
        self.transform = AffineCouplingLayer(net_s=net_s, net_t=net_t)

        self.log_det_jacobian = 0

    def forward(self, z):
        z_ = self.act_norm(z) if self.norm else z
        z_ = self.conv(z)
        z_ = self.transform(z_)

        self.log_det_jacobian = self.act_norm.log_det_jacobian + self.conv.log_det_jacobian + self.transform.log_det_jacobian

        return z_

    def inverse(self, x):
        x_ = self.transform.inverse(x)
        x_ = self.conv.inverse(x_)
        x_ = self.act_norm.inverse(x_) if self.norm else x_

        self.log_det_jacobian = self.act_norm.log_det_jacobian + self.conv.log_det_jacobian + self.transform.log_det_jacobian

        return x_


class Affine(Layer):
    """Affine transformation y = e^a * x + b.

    Args:
        dim (int): dimension of input/output data. int
    """
    def __init__(self, dim: int = 2):
        """ Create and init an affine transformation. """
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(self.dim))  # a
        self.shift = nn.Parameter(torch.zeros(self.dim))  # b
        self.log_det_jacobian = 0

    def forward(self, z):
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward tranformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation, shape [batch_size]
        """
        B, D = z.shape
        y = torch.exp(self.log_scale) * z + self.shift
        self.log_det_jacobian = torch.log(torch.prod(torch.exp(self.log_scale), dim=0) * torch.ones(B))

        return y

    def inverse(self, x):
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse tranformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation, shape [batch_size]
        """
        B, D = x.shape
        x = (x - self.shift) / torch.exp(self.log_scale)
        self.log_det_jacobian = torch.log(torch.prod(1/(torch.exp(self.log_scale) + 1e-8), dim=0) * torch.ones(B))

        return x
