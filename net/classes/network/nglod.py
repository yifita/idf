# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math 

import numpy as npxp

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from network.network import Network
import re


class BaseSDF(Network):
    def __init__(self, config):
        #self.pos_enc : bool = False
        self.input_dim = 3
        self.out_dim = 1
        #self.ff_dim : int = 0
        
        super().__init__(config)

        #if self.ff_dim > 0:
        #    self.gauss_matrix = nn.Parameter(torch.randn([self.ff_dim, 3]) * self.ff_width)
        #    self.gauss_matrix.requires_grad_(False)
        #    self.input_dim += (args.ff_dim * 2) - 3

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)

    def forward(self, args):
        gt_sdf = None
        detach = args.get("detach", True)
        coords = args.get("coords").reshape(-1,3)
        
        if args.get("compute_gt", False):
            gt_sdf = self.runner.data.get_sdf(coords.view(-1, 3))

        if (not detach and not coords.requires_grad):
            detach = True
        if detach:
             coords = coords.clone().detach().requires_grad_(True)
        #coords = self.encode(coords)
        preds = self.sdf(coords)
        value = {"sdf":preds[-1],"detached":coords,"gt":gt_sdf}
        for i in range(len(preds)-1):
            value[f"sdf_{i}"] = preds[i]
        return value
 
    def encode(self, x):
        if self.ff_dim > 0:
            x = F.linear(x, self.gauss_matrix)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        
        elif self.pos_enc:
            x = positional_encoding(x)
        return x

    def sdf(self, x, ids=None):
        return None
        

    
class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)
        self.sparse = None

    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,0,0].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,:,0].permute([1,2,0])

        return sample

class nglod(BaseSDF):
    def __init__(self, config):
        
        self.num_lods : int = 1
        self.fdim : int = 32
        self.fsize : int = 4
        self.hidden_dim : int = 128
        self.pos_invariant : bool = False
        self.interpolate : float = None
        self.joint_decoder : bool = True
        self.nglod_checkpoint : str = None
        super().__init__(config)

        self.lod = None
        self.shrink_idx = (self.num_lods - 1)

        

    def _initialize(self):
        ##auto detect parameters based on checkpoints :) 
        if(self.nglod_checkpoint is not None):
            state_dict = torch.load(self.nglod_checkpoint)
            lods = len([  k for k in state_dict.keys() if k.startswith("features") ])
            self.num_lods = lods
            self.fsize = state_dict["features.0.fm"].shape[2]-1
            self.fdim = state_dict["features.0.fm"].shape[1]
            self.hidden_dim = state_dict["louts.0.0.weight"].shape[0]
            self.pos_invariant = self.fdim == state_dict["louts.0.0.weight"].shape[1] # do we concat the position before
            self.joint_decoder = not "louts.1.0.weight"  in state_dict.keys() # we only have one element in louts
            
        self.features = nn.ModuleList([])
        for i in range(self.num_lods):
            self.features.append(FeatureVolume(self.fdim, self.fsize * (2**i)))
    

        self.louts = nn.ModuleList([])


        self.sdf_input_dim = self.fdim
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim

        self.num_decoder = 1 if self.joint_decoder else self.num_lods 

        for i in range(self.num_decoder):
            self.louts.append(
                nn.Sequential(
                    nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1, bias=True),
                )
            )

        if(self.nglod_checkpoint is not None):
            state_dict = torch.load(self.nglod_checkpoint)
            self.load_state_dict(state_dict, strict=True)
        
    def encode(self, x):
        # Disable encoding at this level
        return x
    
    def grow(self):
        if self.shrink_idx > 0:
            self.shrink_idx -= 1

    def sdf(self, x, return_lst=False):
        # Query

        l = []
        samples = []
        
        for i in range(self.num_lods):
            
            # Query features
            sample = self.features[i](x)
            samples.append(sample)
            
            # Sum queried features
            if i > 0:
                samples[i] += samples[i-1]
            
            # Concatenate xyz
            ex_sample = samples[i]
            if not self.pos_invariant:
                ex_sample = torch.cat([x, ex_sample], dim=-1)

            if self.num_decoder == 1:
                prev_decoder = self.louts[0]
                curr_decoder = self.louts[0]
            else:
                prev_decoder = self.louts[i-1]
                curr_decoder = self.louts[i]
            
            d = curr_decoder(ex_sample)

            # Interpolation mode
            if self.interpolate is not None and self.lod is not None:
                
                if i == len(self.louts) - 1:
                    return d

                if self.lod+1 == i:
                    _ex_sample = samples[i-1]
                    if not self.pos_invariant:
                        _ex_sample = torch.cat([x, _ex_sample], dim=-1)
                    _d = prev_decoder(_ex_sample)

                    return (1.0 - self.interpolate) * _l + self.interpolate * d
            
            # Get distance
            else: 
                d = curr_decoder(ex_sample)
                #self.h = samples[i]
                
                # Return distance if in prediction mode
                if self.lod is not None and self.lod == i:
                    return d

                l.append(d)

           
        return l
        
