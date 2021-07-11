import os
from typing import List
import torch

class AllOf:
    def __init__(self, elements):
        self.elements = elements
    def __getattr__(self, attr):
        def on_all(*args, **kwargs):
            for obj in self.elements:
                getattr(obj, attr)(*args, **kwargs)
        return on_all
    def __iter__(self):
        return self.elements.__iter__()
    def __contains__(self, item):
        return item in self.elements
    def __len__(self):
        return len(self.elements)
    def __getitem__(self, key):
        return self.elements[key]
    def __getstate__(self):
        return self.elements
    def __setstate__(self,value):
        self.elements = value

def get_path_for_data(obj, path:str):
    if not (os.path.isabs(path)) and not (os.path.exists(path)):
        return  obj.runner.data_path + path
    return path

def check_weights(state_dict):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print('NaN Values detected in model weight %s.' % k)
        if not torch.isfinite(v).all():
            print('Infinite Values detected in model weight %s.' % k)

def valid_value_mask(tensor: torch.Tensor):
    return torch.isfinite(tensor) & ~torch.isnan(tensor)

def get_class_from_string(cls_str):
    import importlib
    i = cls_str.rfind('.')
    mod = importlib.import_module(cls_str[:i])
    clss = getattr(mod, cls_str[i + 1:])
    return clss

def slice_dict(d: dict, idx: List):
    """ slice all values in a dict to take the idx
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            slice_dict(v, idx)
        else:
            try:
                iter(v)
            except TypeError as te:
                new_d[k] = v
            else:
                new_d[k] = v[idx]
    return new_d
