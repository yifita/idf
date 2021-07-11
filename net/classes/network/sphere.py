from typing import Tuple
import torch
from torch._C import device
import torch.nn.functional as F
from  network.network import Network

class Sphere(Network):

    def __init__(self, config):
        self.radius  = 0.5
        super().__init__(config)

    def _initialize(self):
        radius = self.radius
        delattr(self,"radius")
        self.register_parameter("radius",torch.nn.Parameter(torch.Tensor([radius])))

    def generate_point_cloud(self, n_points, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.rand(n_points,3, device=self.radius.device, dtype=self.radius.dtype)-0.5
        coords = F.normalize(coords, dim=-1) * self.radius
        normals = F.normalize(coords.clone(), dim=-1)
        return coords, normals

    def encode(self, *args, **kwargs):
        pass

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)

    def forward(self, args):
        detach = args.get("detach", True)
        input_points = args["coords"]
        if detach:
            input_points = input_points.clone().detach().requires_grad_(True)
        result = (input_points.norm(dim=-1) - self.radius)
        return {"sdf":result, "detached":input_points}

    def save(self, path):
        torch.save(self, path)


