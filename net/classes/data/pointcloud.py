import torch
import numpy as np

from helper import get_path_for_data
from data.dataset import CDataset
import pymeshlab


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

class Pointcloud(CDataset):
    """
    Attributes:
        num_points: load total of N points from a dense point cloud
        batch_size: total number of on-and-off-surface query points for training SDF
        pointcloud_size: number of on-surface points for encoding
        keep_aspect_ratio: normalization
        factor_off_surface: ratio of off-surface points
    """
    def __init__(self, config):
        self.path : str = None
        self.pointcloud_path : str = None
        self.num_points : int = 1000000
        self.batch_size : int = 100000
        self.pointcloud_size: int = 3000
        self.keep_aspect_ratio : bool = True
        self.factor_off_surface : float = 0.5
        self.bbox_size : float = 2.0
        self.padding : float = 0.1

        self._has_init : bool = False
        self._coords  : np.ndarray = None
        self._normals : np.ndarray = None

        super().__init__(config)
        if self.pointcloud_path is None:
            self.pointcloud_path = self.path

    def _init(self):
        if(self._has_init):
            return
        path = get_path_for_data(self, self.path)
        self.runner.py_logger.info(f"Loading pointcloud from {path}\n")
        self._coords, self._normals = load_point_cloud(path, self.num_points)
        self._coords = torch.from_numpy(self._coords)
        self._normals = torch.from_numpy(self._normals)
        self.normalize(self._coords, self._normals)
        if self.pointcloud_path != self.path:
            path = get_path_for_data(self, self.pointcloud_path)
            self.runner.py_logger.info(f"Loading pointcloud points from {path}\n")
            pcl_coords, pcl_normals = load_point_cloud(path)
            pcl_coords = torch.from_numpy(pcl_coords)
            pcl_normals = torch.from_numpy(pcl_normals)
            self.normalize(pcl_coords, pcl_normals)
            self.pcl = torch.cat([pcl_coords, pcl_normals], dim=-1)
        else:
            self.pcl = torch.cat([self._coords, self._normals], dim=-1)

        self.runner.logger.log_mesh("pointclouddata", self._coords[None,...], None, vertex_normals=self._normals[None,...])
        self._has_init = True

    def __len__(self):
        self._init()
        return (self._coords.shape[0] // int(self.batch_size*(1-self.factor_off_surface))) + 1

    def __getitem__(self, idx):
        self._init()
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples
        # Random coords
        rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                     size=on_surface_samples))

        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]

        sdf = torch.zeros((total_samples, 1))  # on-surface = 0

        pointcloud = torch.zeros((self.pointcloud_size, 3))

        if self.pointcloud_size > 0:
            # sample randomly from point cloud
            rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                     size=self.pointcloud_size))
            pointcloud = self.pcl[rand_idcs, :]


        if(off_surface_samples > 0):
            off_surface_coords = torch.rand(off_surface_samples, 3) - 0.5
            off_surface_normals = torch.ones((off_surface_samples, 3)) * -1

            off_surface_coords *= self.bbox_size
            sdf[on_surface_samples:, :] = -1  # off-surface = -1

            coords = torch.cat((on_surface_coords, off_surface_coords),
                                    dim=0)
            normals = torch.cat((on_surface_normals, off_surface_normals),
                                    dim=0)

        else:
            coords = on_surface_coords
            normals = on_surface_normals

        return {
                "coords" : coords,
                "normal_out" : normals,
                "sdf_out" : sdf,
                "pointcloud": pointcloud,
                }

