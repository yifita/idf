from network.network import Network
from network.siren import gradient 
import torch
import torch.nn.functional as F
from network.fe.feature_extractor import FeatureExtractor

class GroundTruth(Network):
    
    def __init__(self, config):
        self.finite_grad : bool = True
        super().__init__(config)
       
    def f(self,x):
        return self.runner.data.get_sdf(x.view(-1, 3))
    
    def _initialize(self):
        pass

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)

    def forward(self, args):
        coords = args.get("coords").reshape(-1,3)
        gt_sdf = self.f(coords)
        if(self.finite_grad):
            #We dont care about super accurate numbers here see it as the central  difference of coords+eps/2 & we save 3 computations 
            eps = 1e-4
            eps_x = torch.tensor([[eps, 0.0, 0.0]], device=coords.device)
            eps_y = torch.tensor([[0.0, eps, 0.0]], device=coords.device)
            eps_z = torch.tensor([[0.0, 0.0, eps]], device=coords.device)

            grad = torch.cat([gt_sdf- self.f(coords - eps_x),
                               gt_sdf- self.f(coords - eps_y),
                               gt_sdf - self.f(coords - eps_z)], dim=-1)
            grad = grad / eps
        
        return {"sdf":gt_sdf,"normal":grad,"detached":coords}


        
    

