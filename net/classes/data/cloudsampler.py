from math import ceil
import os
import torch
import numpy as np
from torch._C import dtype
from helper import get_path_for_data
from data.dataset import CDataset
from pykdtree.kdtree import KDTree
import torch
import pymeshlab

""" Sample 3D points and estimate their SDF values from an oriented point cloud """

def load_point_cloud(path, num_points):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    coords = mesh.vertex_matrix().astype('float32')
    normals = mesh.vertex_normal_matrix().astype('float32')
    if num_points < coords.shape[0]:
        idx = np.random.permutation(coords.shape[0])[:num_points]
        coords = np.ascontiguousarray(coords[idx])
        normals = np.ascontiguousarray(normals[idx])
    return coords, normals


class CloudSampler(CDataset):

    def __init__(self, config):
        self.path : str = None
        self.num_points : int = 1000000
        self.batch_size : int = 5000
        self.bbox_size : float = 2.0
        self.factor_off_surface = 0.5
        self.keep_aspect_ratio : bool = True
        self.pointcloud_size: int = 5000
        self.project_points : int = 10
        self.padding : float = 0.1
        self.sample_sdf : bool = True
        self.sample_close_to_surface : bool = False
        self.cache_file : str = None
        super().__init__(config)

        self._has_init : bool = False
        self._coords  : np.ndarray = None
        self._normals : np.ndarray = None
        if self.cache_file is not None:
            self._cache_file = os.path.join(os.path.dirname(self.path), self.cache_file)
        else:
            self._cache_file = os.path.join(os.path.dirname(self.path),
                                        os.path.splitext(os.path.basename(self.path))[0]+
                                        f'_points_bbox_{self.bbox_size}_pad_{self.padding}.npz')

        self._kdtree = None

    def _init(self):
        if self._has_init:
            return
        self.path = get_path_for_data(self,self.path)
        self.runner.py_logger.info(f"Loading pointcloud from {self.path}\n")
        self._coords, self._normals = load_point_cloud(self.path, self.num_points)
        self._coords, self._normals = torch.from_numpy(self._coords), torch.from_numpy(self._normals)
        self.normalize(self._coords, self._normals)
        self._has_init = True

        self.runner.py_logger.info(f"Pointcloud loaded with shape {self._coords.shape}\n")
        self.runner.logger.log_mesh("pointcloud_data", self._coords[None,...], None, vertex_normals=self._normals[None,...])
        self._kdtree = KDTree(self._coords.numpy())

        if self.sample_sdf:
            # sample sdf values
            if os.path.isfile(self._cache_file):
                self.runner.py_logger.info(f'Loading SDF samples from {self._cache_file}')
                with np.load(self._cache_file) as f:
                    self._off_sdf = f['off_sdf']
                    self._off_samples = f['off_sample']
                assert(self._off_samples.shape[0] == self._off_sdf.shape[0])
                self.runner.py_logger.info(f'Loaded {self._off_samples.shape[0]} SDF samples from {self._cache_file}')
            else:
                #
                num_samples = min(1000000,10*self.num_points)
                self.runner.py_logger.info(f'Sampling {num_samples} SDF')
                self._off_samples = np.random.uniform(-0.5, 0.5,
                                                    size=(num_samples, 3)).astype(np.float32) * self.bbox_size
                self._off_sdf, _ = self._get_sdf(self._off_samples)
                np.savez_compressed(self._cache_file,
                    off_sdf=self._off_sdf, off_sample=self._off_samples)
                self.runner.py_logger.info(f"Saved sampled SDF to cache {self._cache_file}")
            self._off_sdf = self._off_sdf.reshape(-1,1)


    def __len__(self):
        self._init()
        return ceil(self._coords.shape[0] / (self.batch_size*(1-self.factor_off_surface)))

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

    def get_sdf_(self, query_points):
        self._init()
        shp = query_points.shape
        is_tensor = torch.is_tensor(query_points)
        if is_tensor:
            device = query_points.device
            dtype = query_points.dtype
            query_points = query_points.cpu().detach().numpy().reshape(-1, 3)

        sdf,normals = self._get_sdf(query_points)
        sdf = sdf.reshape(shp[:-1])
        if is_tensor:
            sdf = torch.from_numpy(sdf).to(device=device, dtype=dtype)
            normals = torch.from_numpy(normals).to(device=device, dtype=dtype)

        return sdf,normals

    def _get_sdf(self, query_points):
        distances, indices = self._kdtree.query(query_points, k=self.project_points)
        distances = distances.astype(np.float32)
        closest_points = self._kdtree.data.reshape(-1, 3)[indices]
        direction_to_surface = query_points[:, np.newaxis, :] - closest_points
        #[-1,0,1] sign map for every normal
        inside_full = np.sign(np.einsum('ijk,ijk->ij', direction_to_surface, self._normals.numpy()[indices]))
        inside = np.sign(np.sum(inside_full, axis=1))
        # as [-1*-1 >0 and 1*1 >0 ] these will be -1 only in case the normals do not align
        # this could be zero if we sample really close in that case it doesn't matter much
        wrong_normals =inside_full*inside[:,np.newaxis]
        normal_idx=np.argmax(wrong_normals,axis=1)
        normals = self._normals.numpy()[indices,:][np.indices([normal_idx.shape[0]]),normal_idx,:]
        #normals = direction_to_surface[0]
        #normals = direction_to_surface/np.linalg.norm(direction_to_surface,ord=2,keepdims=True,axis=1)
        distances = distances[:, 0]
        distances[inside<0] *= -1

        return distances, normals.reshape(-1,3)


    def __getitem__(self, idx):
        self._init()
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples

        rand_idcs = np.random.choice(point_cloud_size,
                                     size=on_surface_samples)

        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]

        sdf = np.zeros((total_samples, 1), dtype=np.float32)  # on-surface = 0

        if off_surface_samples > 0:
            if self.sample_sdf:
                rnd_idx = np.random.choice(self._off_samples.shape[0], off_surface_samples)
                off_surface_coords = self._off_samples[rnd_idx, :]
                sdf[on_surface_samples:, :] = self._off_sdf[rnd_idx, :]
            else:
                if self.sample_close_to_surface:
                    rnd_idx = np.random.choice(point_cloud_size, off_surface_samples)
                    off_surface_coords = self._coords[rnd_idx, :]
                    off_surface_coords += torch.randn_like(off_surface_coords) * 0.1
                else:
                    off_surface_coords = np.random.uniform(-0.5, 0.5, size=(off_surface_samples, 3)) * self.bbox_size

                sdf[on_surface_samples:, :] = -1  # off-surface = -1

            off_surface_normals = np.ones((off_surface_samples, 3), dtype=np.float32) * -1

            coords = np.concatenate((on_surface_coords, off_surface_coords),
                                    axis=0)
            normals = np.concatenate((on_surface_normals, off_surface_normals),
                                    axis=0)
        else:
            coords = on_surface_coords
            normals = on_surface_normals


        pointcloud = np.zeros((self.pointcloud_size, 3), dtype='float32')
        if self.pointcloud_size > 0:
            # sample randomly from point cloud
            rand_idcs = np.random.choice(self._coords.shape[0],
                                         size=self.pointcloud_size)
            pointcloud = self._coords[rand_idcs, :]
            pointcloud_normals = self._normals[rand_idcs, :]
            pointcloud = np.concatenate([pointcloud, pointcloud_normals], axis=-1)

        return {
                "coords" : torch.Tensor(coords).float(),
                "normal_out" : torch.Tensor(normals).float(),
                "sdf_out" : torch.Tensor(sdf).float(),
                'pointcloud': torch.Tensor(pointcloud).float()
                }

