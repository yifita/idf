from network.network import Network
from network.siren import gradient
import torch
import torch.nn.functional as F

class Residual(Network):

    def __init__(self, config):
        self.base : Network = None
        self.residual : Network = None
        self.freeze_base : bool = False
        super().__init__(config)

        self.epoch = 0
        if(self.freeze_base):
             for param in self.base.parameters():
                param.requires_grad = False
        if(hasattr(self.residual,"base")):
            self.residual.base = self.base
    def _initialize(self):
        pass

    def alpha(self,base):
        R = 100
        #den =  (R + torch.exp(base**2 / 0.01))
        #result = (1 + R) / den
        #print(result)
        #return result
        return 100/(base**2/0.0001+100)

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

        grad = gradient(value,input_points_t, graph=True)
        normal = F.normalize(grad,p=2,dim=1)
        residual = self.residual({"coords":input_points_t,"detach":False})["sdf"]
        offset = self.alpha(value)

        prediction = value + (offset*residual)

        self.runner.logger.log_hist("residual",offset*residual)
        self.runner.logger.log_hist("residual",residual)

        return {"sdf":prediction, "detached":input_points_t,"base":value, "residual":offset*residual, "base_normal":normal}

    def set_base(self, base):
        self.base = base
        if(hasattr(self.residual,"base")):
            self.residual.set_base(base)

    def save(self, path):
        torch.save(self, path)
        torch.save(self.base, path[:-4] + "_base" + path[-4:])



