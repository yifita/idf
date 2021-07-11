from .logger import Logger
from torchvision import utils
from torch import Tensor
import os
import trimesh
import numpy as np
import torch
import shutil


def save_pointcloud(filename, points,colors = None, normals=None ):
    save_dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    mesh = trimesh.Trimesh(vertices=points,vertex_normals = normals, vertex_colors = colors)
    test = mesh.visual.kind
    #there is a bug in trimesh of it only saving normals when we tell the exporter explicitly to do so for point clouds.
    #thus we are calling the exporter directly instead of mesh.export(...)
    f = open(filename, "wb")
    data = trimesh.exchange.ply.export_ply(mesh, 'binary',vertex_normal=True)
    f.write(data)
    f.close()
    return

class File(Logger):

    def __init__(self, config):
        self.use_step :bool = True
        super().__init__(config)
        self._task = ""
        self.disable_hist = True

    def _get_file_name(self, prefix, name, ending):
        if(self.use_step):
            return f"{self.log_folder}/{prefix}_{name}_{self.step}.{ending}"
        return  f"{self.log_folder}/{prefix}_{name}.{ending}"

    def _log_config(self, config:str):
        # save config to folder
        with open(os.path.join(self.log_folder, 'config.json'), 'w') as f:
            f.write(config)

    def _log_text(self, name:str, value:str):
       print(f"{name} : {value}")

    def _log_scalar(self, name:str, value : Tensor):
        self.runner.py_logger.info(f"{name} : {value}")

    def _log_image(self, name:str, image : Tensor):

        if(len(image.shape) == 3):
            image = torch.transpose(image,0,2)
        else:
            image -= torch.min(image)
            image /= torch.max(image)
        utils.save_image(image,self._get_file_name("fig", name, "png"))

    def _log_mesh(self, name:str, vertices : Tensor, faces : Tensor, colors : Tensor,
                  vertex_normals: Tensor = None):
        if isinstance(vertices, Tensor):
            vertices = vertices.cpu().numpy()

        if colors is not None:
            if isinstance(colors, Tensor):
                colors = colors.cpu().numpy()
            colors = colors.astype(np.uint8).reshape([-1,3])

        if vertex_normals is not None:
            if isinstance(vertex_normals, Tensor):
                vertex_normals = vertex_normals.cpu().numpy()

            assert(vertex_normals.size == vertices.size)
            vertex_normals = vertex_normals.reshape([-1, 3])

        if(faces is None):
            out_path = self._get_file_name("pts", name, "ply")
            save_pointcloud(out_path, vertices.reshape([-1,3]), colors,
                            normals=vertex_normals)
            self.runner.py_logger.info(f'Saved mesh to {out_path}')
        else:
            if isinstance(faces, Tensor):
                faces = faces.cpu().numpy()
            mesh = trimesh.Trimesh(vertices.reshape([-1,3]),faces.reshape([-1,3]), process=False, vertex_colors=colors)
            out_path = self._get_file_name("mesh", name, "ply")
            mesh.export(out_path)
            self.runner.py_logger.info(f'Saved mesh to {out_path}')

    def _log_hist(self, name:str, values : Tensor):
        if(self.disable_hist):
            return

        if(isinstance(values,Tensor)):
            values = values.detach().cpu().numpy()

        np.savetxt(self._get_file_name("hist", name, "csv"),values, delimiter=',')

    def _log_figure(self, name:str, figure):
        figure.savefig(self._get_file_name("fig", name, "png"))


    def _log_code_base(self):
        code_dir = os.path.join(self.log_folder, 'code')
        if os.path.isdir(code_dir):
            self.runner.py_logger.info(f'Removing existing code base {code_dir}...')
            shutil.rmtree(code_dir)
        shutil.copytree(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), code_dir, symlinks=True)
        self.runner.py_logger.info(f'Copied code base to {code_dir}')