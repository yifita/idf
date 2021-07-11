from typing import Tuple
import torch
from network.network import Network
from helper import get_path_for_data
from network.projectlevelset import sample_uniform_iso_points

class File(Network):
    def __init__(self, config):
        self.path : str = None
        self.network : Network = None
        self.trainable : bool = True
        self.base : Network = None
        super().__init__(config)

    def generate_point_cloud(self, n_points: int, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _initialize(self):
        path = get_path_for_data(self,self.path)
        self.network = torch.load(path)
        while(isinstance(self.network,File)):
            self.network = self.network.network

        self.network.set_runner(self.runner)

        if not self.trainable:
            for param in self.network.parameters():
                param.requires_grad = False

    def forward(self, args):
        return self.network(args)

    def save(self, path):
        self.network.save(path)