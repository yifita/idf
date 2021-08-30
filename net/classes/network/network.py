from collections import OrderedDict
import os
from typing import Tuple,Dict
import numpy as np
from torch.utils import model_zoo
import torch
from checkpoint_io import is_url
from config import ConfigObject
from evaluator.helper import get_surface_high_res_mesh
from runner import Runner
import re


class Network(torch.nn.Module, ConfigObject):

    def __init__(self, config):
        self.name : str = "Network"
        self.state_dict_path : str = ""
        self.ignore_missing_checkpoint : bool = False
        torch.nn.Module.__init__(self)
        ConfigObject.__init__(self, config)
        self._initialize()
        if self.state_dict_path is not None and len(self.state_dict_path) > 0:
            matches = re.findall("\{(.+?)\}", self.state_dict_path)
            matches = list(set(matches))
            repl = {}
            for match in matches:
                new_match = match.replace('.', '-')
                repl[new_match] = str(eval('self.'+match))
                self.state_dict_path = self.state_dict_path.replace(match, new_match)
            self.state_dict_path = self.state_dict_path.format_map(repl)
            self.load_state(self.state_dict_path)

    def _initialize(self):
        raise NotImplementedError

    @property
    def requires_grad(self) -> bool:
        return any([p.requires_grad for p in self.parameters()])

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("runner",None)
        d.pop("key_value",None)
        for k in self.__dict__.keys():
            if k.startswith("_cached_"):
                del d[k]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def load_state(self, path: str) -> None:
        '''
        load states from a path (url or local file).
        Args:
            path (str): address:subkey[:prefix],
                        loaded_state_dict[subkey] points to the module defined by self
        '''

        splitpath = path.split(':')
        if len(splitpath) != 3 and len(splitpath) != 2:
            raise ValueError(f"path must be address:subkey[:topname] ({path})")
        path = splitpath[0]
        subkey = splitpath[1]
        try:
            if is_url(path):
                self.runner.py_logger.info(f"{self.__class__.__name__} loading checkpoint from url {path}")
                state_dict = model_zoo.load_url(path, progress=True)
            elif os.path.isfile(path) or os.path.islink(path):
                self.runner.py_logger.info(f"{self.__class__.__name__} loading checkpoint from file {path}")
                state_dict = torch.load(path)
            else:
                raise FileExistsError(f"Couldn't find eligibal ckpt at {path}")

            for k in subkey.split('.'):
                state_dict = state_dict.get(k, None)
                if state_dict is None:
                    raise ValueError(f"state_dict ({[_ for _ in state_dict.keys()]}) does not contain {k}")

            if state_dict is not None:
                prefix = ''
                if len(splitpath) == 3:
                    prefix = splitpath[2]
                if len(prefix) > 0:
                    # collect subdictionary that starts with prefix
                    state_dict = OrderedDict((k[len(prefix):], v) for k, v in state_dict.items() if k.startswith(prefix))
                    self.runner.py_logger.info(f"Found {len(state_dict)} entries with prefix {prefix}")
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                self.runner.py_logger.info(f"Loaded!")
                if len(missing_keys) > 0:
                    print('Warning: Could not find %s in checkpoint!' % missing_keys)
                if len(unexpected_keys) > 0:
                    print('Warning: Found unexpectedly %s in checkpoint!' % unexpected_keys)

        except Exception as e:
            self.runner.py_logger.error(repr(e))
            #if not self.ignore_missing_checkpoint:
            #    throw object()

    def set_runner(self, runner):
        self.runner = runner
        self.runner.py_logger.info(f"set runner for {type(self).__name__}")
        for m in self.modules():
            m.runner = runner
            self.runner.py_logger.info(f"set runner for {type(m).__name__}")

    def get_class_for(self, name:str, classname:str):
        basename = "network."
        basename += classname.lower()
        return Runner.get_class_for("network", classname)

    def generate_point_cloud(self, n_points : int, data: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        from .projectlevelset import sample_uniform_iso_points
        fea = self.encode(data)

        def sdf_grad_func(coords):
            result = self.evaluate(coords, fea=fea, istrain=False, detach=True)
            sdf, xyz = result['sdf'], result['detached']
            grad = torch.autograd.grad([sdf], [xyz], torch.ones_like(
                sdf), retain_graph=False)[0]
            return sdf, grad

        pointclouds = sample_uniform_iso_points(sdf_grad_func, n_points,
                                                bounding_sphere_radius=1.0)
        return pointclouds.points_packed(), pointclouds.normals_packed()


    def generate_mesh(self, data: dict = None):
        fea = self.encode(data)
        try:
            mesh = get_surface_high_res_mesh(lambda x: self.evaluate(x.unsqueeze(0), fea=fea, detach=True, istrain=False)['sdf'].squeeze(0),
                                             resolution=self.resolution, box_side_length=self.bbox_size, largest_component=False)
            return np.array(mesh.vertices), np.array(mesh.faces), np.array(mesh.vertex_normals)
        except Exception as e:
            self.runner.logger.warn(repr(e))
            return np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float)

    def save(self, path):
        self.runner.error(f"{type(self).__name__} doesn't implement saving")
        pass

    def encode(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


def approximate_gradient(network: Network, query_coords: torch.Tensor,
                         d:float=1e-5, fea:torch.Tensor=None, **kwargs) -> torch.Tensor:
    # calculate steps x + h/2 and x - h/2 for all 3 dimensions
    B, P, _ = query_coords.shape
    # (6, 3)
    step = torch.stack([
        torch.tensor([1., 0, 0]),
        torch.tensor([-1., 0, 0]),
        torch.tensor([0, 1., 0]),
        torch.tensor([0, -1., 0]),
        torch.tensor([0, 0, 1.]),
        torch.tensor([0, 0, -1.])
    ], dim=0).to(device=query_coords.device, dtype=query_coords.dtype) * d / 2
    # B, P, 6, _
    eval_points = query_coords.unsqueeze(2) + step
    f = network.evaluate(eval_points.view(B, 6*P, 3), fea=fea, **kwargs)['sdf'].view(B, P, 6)
    df_dx = torch.stack([
            (f[:, :, 0] - f[:, :, 1]),
            (f[:, :, 2] - f[:, :, 3]),
            (f[:, :, 4] - f[:, :, 5]),
        ], dim=-1) / d
    return df_dx
