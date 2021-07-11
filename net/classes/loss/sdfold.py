from loss.loss import Loss
from loss.siren import gradient
import torch
import torch.nn.functional as F

class SdfOld(Loss):
    def __init__(self, config):
        self.enable_grad = True
        self.sign_loss = False
        self.sdf : float = 3e3
        self.normal : float = 1e2
        self.grad : float = 5e1

        super().__init__(config)
        
    def __call__(self, model_output : dict, model_input: dict):
        '''
        x: batch of input coordinates
        y: usually the output of the trial_soln function
        '''
        gt_sdf = model_input["sdf_out"].flatten()
        gt_normals = model_input["normal_out"]
        coords = model_output["detached"]
        prediction = model_output["sdf"]
        pred_sdf = prediction.flatten()
        sdf_constraint = torch.abs(pred_sdf - gt_sdf)
        result = {"sdf_loss":sdf_constraint}
        
        if(self.enable_grad):
            gradientV = gradient(prediction, coords)

            normal_constraint = 1 - F.cosine_similarity(gradientV,
                                                        gt_normals, dim=-1)
            normal_constraint = torch.where(gt_sdf != 0 ,torch.zeros_like(normal_constraint),normal_constraint)                                            
            grad_constraint = torch.abs(gradientV.norm(dim=-1) - 1)

            result.update({"grad_loss":grad_constraint,"normal_loss":normal_constraint})
            
        if(self.sign_loss):
            sign = model_output["normal"].reshape([-1,3])
            sign_constraint =  1 - F.cosine_similarity(sign,
                                                    gt_normals, dim=-1)
            result["sign_loss"] = sign_constraint
            
        return result
