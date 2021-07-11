from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np
from config import ConfigObject


class CDataset(Dataset, ConfigObject):
    def __init__(self, config):
        self.bbox_size :float = 2.0
        self.padding: float = 0.1
        self.keep_aspect_ratio: bool = True
        self.scaling_range : List = [0.1, 0.1, 0.1]
        self.do_normalize: bool = True
        self.sphere_normalization :bool = False
        Dataset.__init__(self)
        ConfigObject.__init__(self,config)


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def epoch_hook(self, progress):
        pass

    def normalize(self, coords, normals=None):
        if not self.do_normalize:
             return

        if(self.sphere_normalization):
            coord_max = torch.max(coords, axis=0, keepdims=True)[0]
            coord_min = torch.min(coords, axis=0, keepdims=True)[0]
            coord_center = 0.5*(coord_max + coord_min)
            coords -= coord_center
            scale = torch.norm(coords,dim=1).max()
            coords /= scale
            coords *= (self.bbox_size/2 * (1 - self.padding))
            return coord_center, scale

        coord_max = torch.max(coords, dim=0, keepdims=True)[0]
        coord_min = torch.min(coords, dim=0, keepdims=True)[0]
        coord_center = 0.5*(coord_max + coord_min)
        if self.keep_aspect_ratio:
            scale = (coord_max - coord_min).max(dim=-1, keepdims=True)[0]
        else:
            scale = (coord_max - coord_min)

        coords -= coord_center
        coords /= scale
        coords *= (self.bbox_size * (1 - self.padding))

        if normals is not None:
            normals *= scale
            normals /= torch.norm(normals, dim=-1, keepdim=True)

        return coord_center, scale

    def normalize_np(self, coords):
        coord_max = np.max(coords, axis=0, keepdims=True)[0]
        coord_min = np.min(coords, axis=0, keepdims=True)[0]
        coord_center = 0.5*(coord_max + coord_min)
        if self.keep_aspect_ratio:
            scale = (coord_max - coord_min).max(axis=-1, keepdims=True)[0]
        else:
            scale = (coord_max - coord_min)

        coords -= coord_center
        coords /= scale
        coords *= (self.bbox_size * (1 - self.padding))

        return coord_center, scale

