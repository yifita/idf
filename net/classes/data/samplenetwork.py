from math import ceil
from typing import Tuple
from data.dataset import CDataset
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from network.network import Network
from network.siren import gradient
# from network.projectlevelset import sample_uniform_iso_points
from evaluator.helper import get_surface_high_res_mesh


class SampleNetwork(CDataset):

    def __init__(self,config):
        self.num_points: int = 100000
        self.batch_size : int = 5000
        self.factor_off_surface : float = 0
        self.data : CDataset = None
        self.network : Network = None
        self.pointcloud_size: int = 3000
        self.padding : float = 0.1
        self.bbox_size: float = 2.0

        self._has_init : bool = False

        super().__init__(config)
        self._coords : np.array = None
        self._normals : np.array = None
        self.network.requires_grad_(False)

    def _init(self):
        if(self._has_init):
            return

        if torch.cuda.is_available():
            self.network = self.network.cuda()

        if self.data is None:
            self.runner.py_logger.info(f"Initializing {self.__class__.__name__} without input data")
            self._coords, self._normals = self.network.generate_point_cloud(self.num_points, data=None)
            self._coords, self._normals = self._coords.cpu().numpy(), self._normals.cpu().numpy()
            self._coords, self._normals = self.transform(self._coords, self._normals)
            self.runner.logger.log_mesh("samplernetwork_pointcloud", self._coords[None,...], None, vertex_normals=self._normals[None,...])

        self._has_init = True

    def transform(self, coords, normals):
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)


        coord_min = np.array(coord_min).reshape(1,-1)
        coord_max = np.array(coord_max).reshape(1,-1)

        scale = (coord_max - coord_min)
        coords = (coords - coord_min) / scale
        normals /= np.linalg.norm(normals, ord=2, keepdims=True, axis = -1)
        coords -= 0.5
        coords *= (self.bbox_size * (1 - self.padding))
        return coords, normals

    def get_class_for(self, name:str, classname:str):
        #all subclasses are of type network
        return self.runner.get_class_for("network", classname)


    def __len__(self):
        self._init()
        if hasattr(self, '_coords'):
            return ceil(self._coords.shape[0] / int(self.batch_size*(1-self.factor_off_surface)))
        else:
            return len(self.data)

    def __getitem__(self, idx):

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        on_surface_samples = self.batch_size - off_surface_samples
        total_samples = self.batch_size

        data = None
        _coords, _normals = self._coords, self._normals
        if self.data is not None:
            data = self.data[idx]
            _coords, _normals = self.network.generate_point_cloud(self.num_points, data=data)
            self._coords, self._normals = self._coords.numpy(), self._normals.numpy()
            _coords, _normals = self.transform(self._coords, self._normals)

        rand_idcs = np.random.choice(self._coords.shape[0], size=on_surface_samples)
        on_surface_coords = _coords[rand_idcs, :]
        on_surface_normals = _normals[rand_idcs, :]

        sdf = torch.zeros(total_samples, 1)  # on-surface = 0
        if off_surface_samples > 0:
            off_surface_coords = torch.rand(off_surface_samples, 3)*2 -1

            # calc normals
            base = self.network({"coords":off_surface_coords, "detach":True})
            value = base["sdf"]
            projection = base["detached"]
            grad = gradient(value, projection, graph=False).detach()
            off_surface_normals = F.normalize(grad, dim=-1)

            sdf[on_surface_samples:,0] = value

            coords = torch.cat((on_surface_coords, off_surface_coords), axis=0)
            normals =torch.cat((on_surface_normals, off_surface_normals), axis=0)
        else:
            coords = on_surface_coords
            normals = on_surface_normals

        pointcloud = np.zeros((self.pointcloud_size, 3), dtype='float32')

        if self.pointcloud_size > 0:
            # sample randomly from point cloud
            rand_idcs = np.random.choice(_coords.shape[0],
                                     size=self.pointcloud_size)
            pointcloud = _coords[rand_idcs, :]
            pointcloud_normals = _normals[rand_idcs, :]
            pointcloud = np.concatenate([pointcloud, pointcloud_normals], axis=-1)

        return {
                "coords" : coords,
                "normal_out" : normals,
                "sdf_out" : sdf,
                "pointcloud": pointcloud
                }