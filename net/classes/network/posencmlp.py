from collections import defaultdict
from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from network.network import Network

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(in_dim, multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': in_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class PosEncMLP(Network):
    '''
    Based on: https://github.com/facebookresearch/DeepSDF
    and https://github.com/matanatz/SAL/blob/master/code/model/network.py
    '''
    def __init__(self, config):
        # default
        self.dim: int = 3
        self.c_dim : int = 0
        self.hidden_size: int = 512
        self.n_layers: int = 8
        self.bias: float = 0.6
        self.weight_norm: bool = False
        self.skip_in: List =[]
        self.num_frequencies: int = 6
        self.geometric_init: bool = True
        self.annealing : bool = True
        self.annealing_params : Dict = None
        self.has_last_nonlinear: bool = True
        self.out_dim : int = 1
        super().__init__(config)
        # update annealing params with default ones
        self._annealing_params = dict(maxing_step=0.8)
        if self.annealing_params is None:
            self.annealing_params = {}
        for k, v in self._annealing_params.items():
            if k not in self.annealing_params:
                self.annealing_params[k] = v

        self._epoch = 0
        self._progress = 0

    def _initialize(self):
        # shortcuts
        dim = self.dim
        hidden_size = self.hidden_size
        n_layers = self.n_layers
        num_frequencies = self.num_frequencies
        skip_in = self.skip_in
        bias = self.bias
        weight_norm = self.weight_norm

        dims = [dim] + [hidden_size] * n_layers + [self.out_dim]

        self.embed_fn = None
        if num_frequencies > 0:
            embed_fn, input_ch = get_embedder(dims[0], num_frequencies)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        dims[0] += self.c_dim
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - (dims[0] - self.c_dim)
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(
                        np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, self.dim:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :self.dim], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif num_frequencies > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - self.dim):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def annealing_encoding_weight(self, progress:float) -> torch.Tensor:
        """ returns 2K+1 tensor"""
        max_freq = self.num_frequencies-1
        freq_bands = torch.linspace(0., max_freq, self.num_frequencies)
        alpha = (progress/self.annealing_params['maxing_step'])*max_freq
        omega_j = (1-torch.cos(torch.clamp(alpha - freq_bands, 0, 1)*np.pi))/2
        self.runner.logger.log_scalar("annealing",
            dict([("k=%d" % k.item(), v) for v, k in zip(omega_j.unbind(), freq_bands.unbind())]))
        omega_j = torch.repeat_interleave(omega_j, 2*self.dim)
        omega_j = torch.cat([torch.tensor([1]*self.dim), omega_j])
        assert(omega_j.nelement() == self.num_frequencies*2*self.dim + self.dim)
        return omega_j

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)

    def forward(self, args):
        is_train = args.get('istrain', self.training)
        coords, c, detach = args['coords'], args.get('x', None), args.get('detach', True)
        self._epoch = args.get('epoch', self._epoch)
        self._progress = args.get('progress', self._progress)
        if detach:
            # allows to take derivative w.r.t. input
            coords = coords.clone().detach().requires_grad_(True)

        outputs = {"detached": coords}
        if self.embed_fn is not None:
            coords = self.embed_fn(coords)
            if is_train and self.annealing:
                annealing = self.annealing_encoding_weight(self._progress).to(device=coords.device)
                coords = coords*annealing

        x = coords
        if c is not None and c.numel() > 0:
            assert(x.ndim == c.ndim)
            x = torch.cat([x, c], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, coords], -1) / np.sqrt(2)

            x = lin(x)

            if self.has_last_nonlinear and l < self.num_layers - 2:
                x = self.softplus(x)

        outputs["sdf"] = x
        return outputs
