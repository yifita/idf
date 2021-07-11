import torch
from  network.network import Network
from torch import nn
import numpy as np


def gradient(y, x, grad_outputs=None, graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,retain_graph=True,
                               create_graph=graph)[0]
    return grad


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussi
    # on of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies t
    # he activations before the
    # nonlinearity. Different signals may require different omega_0 in the first
    #  layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to
    # keep the magnitude of
    # activations constant, but boost gradients to the weight matrix
    # (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.dim = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.dim,
                                            1 / self.dim)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.dim) / self.omega_0,
                                            np.sqrt(6 / self.dim) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class SirenModule(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, detach=True):
        if(detach):
            # allows to take derivative w.r.t. input
            coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords



class Siren(Network):

    def __init__(self, config):
        self.omega :  float = 15.0
        self.hidden_size : int = 256
        self.hidden_layers : int = 3
        self.dim : int = 3
        self.out_dim : int = 1
        self.outermost_linear : bool = True
        self._module : nn.Module = None
        super().__init__(config)

    def _initialize(self):
        self._module = SirenModule(in_features=self.dim, out_features=self.out_dim,
                        hidden_features=self.hidden_size, hidden_layers=self.hidden_layers,
                        outermost_linear=self.outermost_linear,
                        first_omega_0=self.omega,
                        hidden_omega_0=self.omega)

    def forward(self, args):
        detach = args.get("detach",True)
        input_coords = args["coords"]
        c = args.get("x", None)
        if c is not None:
            input_coords = torch.cat([input_coords, c], dim=-1)
        result, detached = self._module(input_coords, detach)

        return {"sdf":result, "detached":detached}

    def evaluate(self, coords, fea=None, **kwargs):
        kwargs.update({'coords': coords, 'x': fea})
        return self.forward(kwargs)

    def save(self, path):
        torch.save(self, path)
