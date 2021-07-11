import sys
from typing import Dict
import numpy as np
import torch
from evaluator.evaluator import Evaluator
from .helper import get_surface_high_res_mesh
from task.chamfer import Compute_Chamfer
import trimesh

class Mesh(Evaluator):

    def __init__(self, config):
        self.batch_size = 100000
        self.frequency =  10
        self.resolution = 100
        self.offset = 0
        self.skip_first :bool = True
        self.bbox_size : float = 2.0
        self.attribute : str = "sdf"
        self.stop_after : int = 0
        self.compute_chamfer : bool = False
        super().__init__(config)
        self.attributes.append(self.attribute)

    def evaluate_mesh_value(self, data=None):
        fea = self.encode_network(data)
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size

        data.pop('coords', None)
        try:
            mesh = get_surface_high_res_mesh(lambda x: self.evaluate_network(x.unsqueeze(0), fea=fea, **data)[self.attribute].squeeze(0),
                resolution=self.resolution, box_side_length=bbox_size, largest_component=False)
            if not mesh.is_empty:
                return np.array(mesh.vertices), np.array(mesh.faces), np.array(mesh.vertex_normals)
            return np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float)
        except Exception as e:
            tb = sys.exc_info()[2]
            self.runner.py_logger.error(repr(e))

            return np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float)


    def epoch_hook(self, epoch, data: Dict =None):
        if (epoch % self.frequency == 0 and (epoch <= self.stop_after or self.stop_after == 0)):
            if self.runner.active_task.__class__.__name__ == "Train" and self.skip_first and epoch == 0:
                return
            self.runner.py_logger.info(f"Generating {self.name} with resolution {self.resolution}")
            #this is plain wrong
            #with torch.no_grad():
            verts, faces, _ = self.evaluate_mesh_value(data)
            if(self.compute_chamfer):
                Compute_Chamfer(trimesh.Trimesh(verts.astype(np.float32),faces),self.runner,f"{self.name}_chamfer_{{0}}_{{1}}", True)
            self.runner.logger.log_mesh(self.name, verts.reshape([1, -1, 3]),
                                        faces.reshape([1, -1, 3]))
