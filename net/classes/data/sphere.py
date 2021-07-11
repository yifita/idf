from data.dataset import CDataset
import numpy as np
import torch 

class Sphere(CDataset):

    def __init__(self,config):
        self.radius :int = 0.5
        self.batch_size : int = 100000
        self.num_points : int = 1000000
        self._coord_min = np.array([0,0,0]).reshape(1,-1)
        self._coord_max = np.array([1,1,1]).reshape(1,-1)
        super().__init__(config)
        #for some networks we need a pointcloud so sample some 
        coords = ((torch.randn(self.num_points,3)*2)-1).float()
        self._normals = (coords/coords.norm(dim = 1,keepdim = True)).numpy()
        self._coords = (self.radius * self._normals)
        
    def __len__(self):
        return (self.num_points// self.batch_size) + 1
     
    def __getitem__(self, idx):
        coords = ((torch.rand(self.batch_size,3)*2)-1).float()
        normals = coords 
        sdf = coords.norm(dim=1) - self.radius
        return {"coords":torch.Tensor(coords), "sdf_out":torch.Tensor(sdf),"normal_out":torch.Tensor(normals)}