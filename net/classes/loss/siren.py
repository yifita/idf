from network.network import Network, approximate_gradient
from network.siren import gradient
from loss.loss import Loss
import torch.nn.functional as F
import torch


class Siren(Loss):

    def __init__(self, config):
        self.sdf : float = 3e3
        self.inter : float = 1e2
        self.normal : float = 1e2
        self.grad : float = 5e1
        super().__init__(config)


    def __call__(self, model_output : dict, model_input: dict):
        '''
        x: batch of input coordinates
        y: usually the output of the trial_soln function
        '''

        gt_sdf = model_input["sdf_out"].reshape([-1])
        gt_normals = model_input["normal_out"].reshape([-1,3])
        pred_sdf = model_output["sdf"].reshape([-1])
        coords = model_output["detached"]

        approximate_gradient = getattr(self.runner.network, 'approximate_gradient', False)
        if approximate_gradient:
            fea = model_output.get('encoded', None)
            gradientV = approximate_gradient(self.runner.network, coords.detach(), fea=fea)
        else:
            gradientV = gradient(pred_sdf, coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a
        # boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf,
                                    torch.zeros_like(pred_sdf)) * self.sdf

        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf),
                                        torch.exp(-1e2 * torch.abs(pred_sdf))) * self.inter
        normal_constraint = torch.abs(torch.where(gt_sdf != -1,
                                        1 - F.cosine_similarity(gradientV,
                                                                gt_normals, dim=-1),
                                        torch.zeros_like(gt_sdf))) * self.normal
        grad_constraint = torch.abs(gradientV.norm(dim=-1) - 1) * self.grad

        return  {"sdf_loss":torch.abs(sdf_constraint),
                 "inter_loss": inter_constraint,
                 "normal_loss":normal_constraint,
                 "grad_loss":grad_constraint}