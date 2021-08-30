from network.network import Network
from pykdtree.kdtree import KDTree
from network.siren import gradient
import torch
import torch.nn.functional as F
from network.fe.feature_extractor import FeatureExtractor

class Residual(Network):

    def __init__(self, config):
        self.base : Network = None
        self.residual : Network = None
        self.freeze_base : bool = False
        self.close_surface_activation : bool = False
        self.activation_threshold : float = 0.02
        self.use_tanh : bool = False
        self.alpha : float = 0.05
        super().__init__(config)

        self.epoch = 0
        if(self.freeze_base):
             for param in self.base.parameters():
                param.requires_grad = False
        if(hasattr(self.residual,"base")):
            self.residual.base = self.base

    def _initialize(self):
        pass

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)


    def forward(self, args):
        is_train = args.get("istrain",False)
        if(is_train):
            self.epoch += 1
        detach = args.get("detach",True)
        input_points_t = args.get("coords",None).reshape(-1,3)
        if(not input_points_t.requires_grad):
            detach = True

        coords = input_points_t

        base = self.base({"coords":input_points_t,"detach":detach})
        value = base["sdf"]
        input_points_t = base["detached"]

        if self.close_surface_activation:
            activation = 1.0 / (1 + (value.detach()/self.activation_threshold)**4)
        else:
            activation = torch.ones_like(value).detach()


        grad = gradient(value,input_points_t, graph = True)
        normal = F.normalize(grad,p=2,dim=1)
        residual = self.residual({"coords":input_points_t,"detach":False})["sdf"]

        if self.use_tanh:
            residual = torch.tanh(residual)
        residual = self.alpha*residual*activation
        prediction = value + residual

        return {"sdf":prediction, "detached":input_points_t,"base":value, "residual":residual, "base_normal":normal}

    def set_base(self, base):
        self.base = base
        if(hasattr(self.residual,"base")):
            self.residual.set_base(base)

    def save(self, path):
        torch.save(self, path)
        torch.save(self.base, path[:-4] + "_base" + path[-4:])



