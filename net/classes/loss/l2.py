from loss.loss import Loss
from network.siren import gradient

import torch

class L2(Loss):
    
    def __init__(self, config):
        self.pre_factor : float = 1
        super().__init__(config)


    def __call__(self, model_output : dict, model_input: dict):
        pred = model_output["sdf"].reshape(-1,1)
        input = model_output["detached"]
        true_sdf = model_input["sdf_out"].reshape(-1,1)
        loss = torch.nn.functional.mse_loss(self.pre_factor * pred,self.pre_factor * true_sdf,reduction='none')
        loss = {"sdf_loss":loss}
        return loss
        