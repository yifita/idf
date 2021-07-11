from typing import Dict
from evaluator.evaluator import Evaluator
import torch
from pykdtree.kdtree import KDTree
from network.siren import gradient
import numpy as np


class Chamfer(Evaluator):

    def __init__(self, config):
        self.attributes = ["sdf"]
        self.batch_size = 75000
        self.step_optim = 5
        self.use_random = False
        self.random_size = 1000000
        self.frequency = 10
        super().__init__(config)
        self.init = False
        self.data = None
        self._kdtree = None
        self.batches = 0
        self.length = 0

    def _init(self):
        if(self.init):
            return
        len(self.runner.data)
        if(self.use_random):
            self.data = np.random.uniform(-1,1,[self.random_size,3])
        else:
            self.data = self.runner.data._coords.clone().detach().cpu().numpy()

        self.length = self.data.shape[0]
        self.batches = int((self.length+self.batch_size-1)/self.batch_size)

        self._kdtree:KDTree = KDTree(self.data)

    def project(self,attr, coords, fea):

        input_points_t = torch.Tensor(coords).float().cuda()
        coords = input_points_t
        projection = coords

        for i in range(self.step_optim):
            base = self.runner.network.evaluate(projection, fea=fea, detach=True)
            value = base["sdf"]
            projection = base["detached"]
            grad = gradient(value,projection, graph = False).detach()
            normal = grad/(1e-6 + torch.norm(grad,dim=1,keepdim=True))
            projection = projection - value*normal

        return projection.detach().cpu().numpy(),normal.detach().cpu().numpy()


    def evaluate(self, attr, fea):
        if(self._kdtree == None):
            self._init()
        #init pointcloud
        output = np.zeros([self.length,3], dtype=self.data.dtype)
        normal =  np.zeros([self.length,3], dtype=self.data.dtype)
        for i in range(self.batches):
            imin = i*self.batch_size
            imax = min(imin+self.batch_size,self.length)
            subset = self.data[imin:imax,:]
            output[imin:imax,:],normal[imin:imax,:]=self.project(attr,subset, fea)

        dist,_ = self._kdtree.query(output)
        inverseTree = KDTree(output)
        if(self.use_random):
            inverseDist,_ = inverseTree.query(self.runner.data._coords.clone().detach().cpu().numpy())
        else:
            inverseDist,_ = inverseTree.query(self.data)

        for k,v in [("dist_1",dist),("dist_2",inverseDist),("dist_sum",dist+inverseDist)]:
            self.runner.logger.log_hist(f"{self.name}_{attr}_{k}_hist",v)
            self.runner.logger.log_scalar(f"{self.name}_{attr}_{k}_scalar",np.mean(v))
        self.runner.py_logger.info(f"CHAMFER DISTANCE {np.mean(dist+inverseDist)}")
        self.runner.logger.log_mesh(f"{self.name}_{attr}_pc", output.reshape(1,-1,3), None, vertex_normals=normal.reshape(1,-1,3))
        

    def epoch_hook(self, epoch, data: Dict=None):
        if(epoch % self.frequency == 0 and epoch > 0):
            self.runner.py_logger.info(f"Computing Chamfer distance")
            fea = self.encode_network(data)
            for attr in self.attributes:
                self.evaluate(attr,fea)

