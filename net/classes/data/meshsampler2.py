from math import ceil
import torch
import numpy as np
from helper import get_path_for_data
from data.dataset import CDataset
import torch
import trimesh
from pysdf import SDF
import matplotlib.colors as colors
import matplotlib.cm as cmx

class MeshSampler2(CDataset):

    def __init__(self, config):
        self.path : str = None
        self.num_points : int = 1000000
        self.batch_size : int = 5000
        self.bbox_size : float = 2.0
        self.factor_off_surface = 0.5
        self.keep_aspect_ratio : bool = True
        self.project_points : int = 10
        self.padding : float = 0
        self.flip_y : bool = False
        self.flip_normal : bool = False
        self.sphere_normalization : bool = False
        self.factor_close_surface : float = 0
        self.sample_sdf : bool = True
        self.resample : bool = True
        super().__init__(config)

        self._has_init : bool = False
        self._coords  : np.ndarray = None
        self._normals : np.ndarray = None
        self.sdf = None


    def _init(self):
        if self._has_init:
            return
        self.path = get_path_for_data(self,self.path)
        self.runner.py_logger.info(f"Loading mesh from {self.path}\n")
        mesh = trimesh.load(self.path)
        if(self.flip_normal):
            mesh.faces[:,:] = mesh.faces[:,::-1]

        if(self.sphere_normalization):
            coords = mesh.vertices
            coord_max = np.max(coords, axis=0, keepdims=True)[0]
            coord_min = np.min(coords, axis=0, keepdims=True)[0]
            coord_center = 0.5*(coord_max + coord_min)
            coords -= coord_center
            scale = np.linalg.norm(coords,axis=1).max()
            coords /= scale
            coords *= (self.bbox_size/2 * (1 - self.padding))
        else:
            self.normalize_np(mesh.vertices)

        if(self.flip_y):
            mesh.vertices[:,1] *= -1

        self.sdf = SDF(mesh.vertices,mesh.faces)
        self.mesh = mesh
        self.sample()
        self._has_init = True

    def __len__(self):
        self._init()
        return ceil(self.num_points / (float(self.batch_size)*(1-self.factor_off_surface)))

    def get_sdf(self, query_points):
        self._init()
        shp = query_points.shape
        is_tensor = torch.is_tensor(query_points)
        if is_tensor:
            device = query_points.device
            dtype = query_points.dtype
            query_points = query_points.cpu().detach().numpy().reshape(-1, 3)

        sdf = self._get_sdf(query_points)[0]
        sdf.reshape(shp[:-1])
        if is_tensor:
            sdf = torch.from_numpy(sdf).to(device=device, dtype=dtype)

        return sdf

    def _get_sdf(self, query_points):
        distances = -self.sdf(query_points)
        return distances, None


    def get_color(self,values, vmin = -1,vmax = 1,cmap="seismic"):
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        cMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        return cMap.to_rgba(values,bytes=True)[:,:3]


    def sample(self):
        off_surface_samples = int(self.num_points  * self.factor_off_surface)
        close_surface_samples = int(self.num_points  * self.factor_close_surface)
        on_surface_samples = self.num_points - off_surface_samples - close_surface_samples
        save_mesh =self._coords is None

        self._coords, f_idx = trimesh.sample.sample_surface(self.mesh, self.num_points)
        self._normals = self.mesh.face_normals[f_idx]
        self._coords, self._normals = torch.from_numpy(self._coords), torch.from_numpy(self._normals)

        if off_surface_samples > 0:
            self.off_surface_coords = np.random.uniform(-0.5, 0.5, size=(off_surface_samples, 3)) * self.bbox_size
            if(self.sample_sdf):
                self.off_surface_sdf = -self.sdf(self.off_surface_coords)
            else:
                self.off_surface_sdf = -np.ones(off_surface_samples)
        if close_surface_samples > 0:
            self.close_surface_coords,unused = trimesh.sample.sample_surface(self.mesh, close_surface_samples)
            self.close_surface_coords += np.random.normal(0,1,size=(close_surface_samples, 3))*0.01
            if(self.sample_sdf):
                self.close_surface_sdf = -self.sdf(self.close_surface_coords)
            else:
                self.close_surface_sdf = -np.ones(close_surface_samples)

        if (save_mesh):
                self.runner.logger.log_mesh("pointcloud_data", self._coords[None,...], None, vertex_normals=self._normals[None,...])
                self.runner.logger.log_mesh("pointcloud_data_close", self.close_surface_coords[None,...], None, vertex_normals=np.ones((1,close_surface_samples,3)), colors=self.get_color(self.close_surface_sdf,-0.015,0.015).reshape(1,-1,3) )
                self.runner.logger.log_mesh("pointcloud_data_random", self.off_surface_coords[None,...], None, vertex_normals=np.ones((1,off_surface_samples,3)), colors=self.get_color(self.off_surface_sdf,-1,1).reshape(1,-1,3) )



    """
    def sample(self):
        save_mesh =self._coords is None
        self._coords, f_idx = trimesh.sample.sample_surface(self.mesh, self.num_points)
        self._normals = self.mesh.face_normals[f_idx]
        self._coords, self._normals = torch.from_numpy(self._coords), torch.from_numpy(self._normals)
        self.runner.py_logger.info(f"Pointcloud sampled with shape {self._coords.shape}\n")
        #prevent disk space explosion
        if(save_mesh):
                self.runner.logger.log_mesh("pointcloud_data", self._coords[None,...], None, vertex_normals=self._normals[None,...])
    """

    def __getitem__(self, idx):
        self._init()
        if idx == 0 and (self.resample or self._coords is None):
            self.sample()

        coords  = []
        sdf = []
        normals = []

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        close_surface_samples = int(self.batch_size  * self.factor_close_surface)
        on_surface_samples = self.batch_size - off_surface_samples - close_surface_samples

        if(on_surface_samples > 0):
            endIdx = min(self.num_points,(idx+1)*on_surface_samples)
            startIdx = endIdx-on_surface_samples
            coords.append(self._coords[startIdx:endIdx, :])
            normals.append(self._normals[startIdx:endIdx, :])
            sdf.append(np.zeros((on_surface_samples), dtype=np.float32))  # on-surface = 0

        if off_surface_samples > 0:
            endIdx = min(self.off_surface_coords.shape[0],(idx+1)*off_surface_samples)
            startIdx = endIdx-off_surface_samples
            coords.append(self.off_surface_coords[startIdx:endIdx, :])
            sdf.append(self.off_surface_sdf[startIdx:endIdx])  # on-surface = 0
            normals.append(np.ones((off_surface_samples, 3), dtype=np.float32) * -1)

        if close_surface_samples > 0:
            endIdx = min(self.close_surface_sdf.shape[0],(idx+1)*close_surface_samples)
            startIdx = endIdx-close_surface_samples
            coords.append(self.close_surface_coords[startIdx:endIdx, :])
            sdf.append(self.close_surface_sdf[startIdx:endIdx])  # on-surface = 0
            normals.append(np.ones((close_surface_samples, 3), dtype=np.float32) * -1)


        coords = np.concatenate(coords,axis=0)
        normals = np.concatenate(normals,axis=0)
        sdf = np.concatenate(sdf,axis=0).reshape([-1,1])
        return {
                "coords" : torch.Tensor(coords).float(),
                "normal_out" : torch.Tensor(normals).float(),
                "sdf_out" : torch.Tensor(sdf).float()
                }

