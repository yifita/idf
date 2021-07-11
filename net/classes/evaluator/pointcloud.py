from evaluator.evaluator import Evaluator
from skimage import measure
import torch
import numpy as np
import open3d as o3d


class Pointcloud(Evaluator):


    def __init__(self, config):
        self.attribute : str = "detached"
        self.batch_size = 100000
        self.frequency =  10
        self.resolution = 40
        self.normals = False

        super().__init__(config)

        N = self.resolution
    def evaluate_mesh_value(self, data):
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 9)
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() // N) % N
        samples[:, 0] = ((overall_index.long() // N) // N) % N
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        torch.cuda.empty_cache()
        chunkCount = np.ceil(samples.shape[0]/self.batch_size)

        for i in range(int(chunkCount)):
            cMin = i*self.batch_size
            cMax = min(samples.shape[0], cMin+self.batch_size)
            modelInputT = samples[cMin:cMax, 0:3].cuda()
            if(self.normals):
                samples[cMin:cMax,3:9] = self.evaluate_network(modelInputT)[self.attribute].detach().squeeze().cpu()
            else:
                samples[cMin:cMax,3:6] = self.evaluate_network(modelInputT)[self.attribute].detach().squeeze().cpu()[:,0:3]

        sdfFull = samples[:,3:6].reshape([-1, 3]).cpu().detach().numpy()
        normals = samples[:,6:9].reshape([-1, 3]).cpu().detach().numpy()
        return sdfFull, normals


    def epoch_hook(self, epoch):
        if(epoch % self.frequency == 0):
            verts, normals = self.evaluate_mesh_value()
            self.runner.py_logger.info(f" vertices {verts.shape}")
            self.runner.logger.log_mesh(self.name,verts.reshape([1, -1, 3]),None)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            if( not self.normals):
                pcd.estimate_normals()
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector(radii))
            #pcd, o3d.utility.DoubleVector(radii))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            self.runner.logger.log_mesh(self.name+"del",verts.reshape([1, -1, 3]),faces.reshape([1,-1,3]))
