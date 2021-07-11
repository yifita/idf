from typing import Union, List
from loss.loss import Loss
from loss.siren import gradient
import torch
import torch.nn.functional as F
from network.network import approximate_gradient


class Sdf(Loss):
    def __init__(self, config):
        self.sign_loss = False
        self.sdf : Union[List, float] = 3e2
        self.normal : Union[List, float] = 2e2
        self.grad : Union[List, float] = 5e1
        self.inter : Union[List, float] = 1e2
        self.base_sdf : Union[List, float] = 0
        self.base_normal : Union[List, float] = 0
        self.base_inter: Union[List, float] = 0
        self.base_grad: Union[List, float] = 0
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
        pred_sdf = prediction.view_as(gt_sdf)

        progress = model_input.get('progress', 1.0)

        sdf = self._get_loss_weight('sdf', progress)
        inter =self._get_loss_weight('inter', progress)

        grad = self._get_loss_weight('grad', progress)
        normal = self._get_loss_weight('normal', progress)

        base_sdf = self._get_loss_weight('base_sdf', progress)
        base_inter = self._get_loss_weight('base_inter', progress)
        base_grad = self._get_loss_weight('base_grad', progress)
        base_normal = self._get_loss_weight('base_normal', progress)

        self.runner.logger.log_scalar("loss_weight",
            dict([("weight_sdf", sdf),
            ("weight_inter", inter),
            ("weight_grad", grad),
            ("weight_normal", normal),
            ("weight_base_sdf", base_sdf),
            ("weight_base_inter", base_inter),
            ("weight_base_grad", base_grad),
            ("weight_base_normal", base_normal)
            ]))

        approx_grad = getattr(self.runner.network, 'approximate_gradient', False)
        fea = model_output.get('encoded', None)

        # SDF on the surface
        result = {}
        if sdf > 0:
            sdf_constraint = torch.where(gt_sdf != -1, (pred_sdf-gt_sdf).abs(),
                                        torch.zeros_like(pred_sdf)) * sdf

            result["sdf_loss"] = sdf_constraint

        # SDF off the surface
        if inter > 0:
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf),
                                        torch.exp(-1e2 * torch.abs(pred_sdf))) * inter
            result["inter_loss"] = inter_constraint

        ############## normal #############
        if grad > 0 or normal > 0:
            if approx_grad:
                gradientV = approximate_gradient(self.runner.network, coords.detach(), fea=fea)
            else:
                gradientV = gradient(prediction, coords)

            # eikonal constraint
            if grad > 0:
                grad_constraint = torch.abs(gradientV.norm(dim=-1) - 1) * grad
                result['grad_loss'] = grad_constraint

            # normal direction for the on-surface points
            gradientV = gradientV.view_as(gt_normals)
            if normal > 0:
                normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                            1 - F.cosine_similarity(gradientV,
                                                                    gt_normals, dim=-1),
                                            torch.zeros(gt_normals.shape[:-1], device=gt_normals.device))) * normal

                result["normal_loss"] = normal_constraint

        ############# base #############
        # base sdf
        if 'base' in model_output:
            prediction_base = model_output['base'].view_as(gt_sdf)
            if base_sdf > 0:
                base_constraint = torch.where(gt_sdf != -1, (prediction_base-gt_sdf).abs(),
                                            torch.zeros_like(prediction_base)) * base_sdf
                result['base_sdf_loss'] = base_constraint

            # SDF off the surface
            if base_inter > 0:
                inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(prediction_base),
                                            torch.exp(-1e2 * torch.abs(prediction_base))) * base_inter
                result["base_inter_loss"] = inter_constraint

        # base normal direction and eikonal
        if base_normal > 0 and 'base_normal' in model_output:
            gradientV = model_output['base_normal'].view_as(gt_normals)
            base_normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    1 - F.cosine_similarity(gradientV,
                                                            gt_normals, dim=-1),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device))) * base_normal
            result['base_normal_loss'] = base_normal_constraint
            # also apply eikonal
            result['base_grad_loss'] = (gradientV.norm(dim=-1) -1).abs() * base_grad

        if(self.sign_loss):
            sign = model_output["normal"].reshape([-1,3])
            sign_constraint =  1 - F.cosine_similarity(sign,
                                                       gt_normals, dim=-1)
            result["sign_loss"] = sign_constraint

        return result