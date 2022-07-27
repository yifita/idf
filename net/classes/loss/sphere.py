from loss.loss import Loss
from loss.siren import gradient
import torch
import torch.nn.functional as F

class Sphere(Loss):
    def __init__(self, config):
        self.enable_grad = True
        self.sign_loss = False
        self.radius : float = None
        super().__init__(config)

    def __call__(self, model_output : dict, model_input: dict):
        '''
        x: batch of input coordinates
        y: usually the output of the trial_soln function
        '''
        gt_sdf = model_input["sdf_out"]
        gt_normals = model_input["normal_out"]
        coords = model_output["detached"]
        prediction = model_output["sdf"]
        sdf_constraint = torch.abs(prediction.reshape(-1) - (coords.norm(dim=-1) - self.radius).reshape(-1))
        result = {"sdf_loss":sdf_constraint}

        if(self.enable_grad):
            gradientV = gradient(prediction, coords)

            normal_constraint = 1 - F.cosine_similarity(gradientV,
                                                        coords, dim=-1)
            grad_constraint = torch.abs(gradientV.norm(dim=-1) - 1)

            result.update({"grad_loss":grad_constraint,"normal_loss":normal_constraint})

        if(self.sign_loss):
            sign = model_output["normal"].reshape([-1,3])
            sign_constraint =  1 - F.cosine_similarity(sign,
                                                    coords, dim=-1)
            result["sign_loss"] = sign_constraint

        return result