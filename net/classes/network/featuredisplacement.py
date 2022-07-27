from typing import Dict, List
from .network import Network
import torch
import torch.nn.functional as F


_VALID_MODES = ("BN", "BSDF", "LC")
class FeatureDisplacement(Network):

    def __init__(self, config):
        self.base : Network = None
        self.residual : Network = None
        self.feature : Network = None
        self.freeze_base : bool = False
        self.offset_base : float = 1.0
        self.offset_max : float = 1.0
        self.pointcloud_sigma: float = 0.0
        self.pointcloud_size: int = 3000
        self.feature_only : bool = False
        self.close_surface_activation: bool = True
        self.activation_threshold: float = 0.01
        self.base_normal_in_c: bool = False  # input base normal to residual net
        self.approximate_gradient: bool = False
        self.scale_base_value: bool = True  # scale base value before inputting to residual net
        self.detach_normal: bool = True
        self.detach_query: bool = False
        self.use_tanh: bool = True
        self.query_modes: List[str] = ["BSDF"]  # BSDF (base_sdf), BN (base_normal), LC (local cell coords)
        # duplicated
        self.base_normal_in_coord: bool = False  # input base normal to residual net
        super().__init__(config)

        for m in self.query_modes:
            if m not in _VALID_MODES:
                self.runner.py_logger.warn(f"Invalid mode ({m}). Supported modes are {_VALID_MODES}")

    def _initialize(self):
        if self.freeze_base and self.base is not None:
             self.base.requires_grad_(False)

        self.register_buffer('factor', torch.tensor(self.offset_max))
        self.epoch = 0

        # compatibility deprecated
        if self.base_normal_in_coord:
            self.query_modes = ["BSDF", "BN"]


    def encode(self, args:Dict)->torch.Tensor:
        force_run = args.get("force_run", False)
        is_train = args.get("istrain", self.training)
        if not force_run and is_train and not self.residual.requires_grad:
            return None

        pointcloud_size = self.pointcloud_size
        try:
            pointcloud_size = self.runner.active_task.pointcloud_size
        except Exception:
            pass

        pointcloud = args['pointcloud']

        if self.pointcloud_sigma > 0.0:
            pointcloud += self.pointcloud_sigma * torch.randn_like(pointcloud)

        if args is not None:
            is_train = args.get("istrain", self.training)
            self.epoch = args.get("epoch", self.epoch)
            if is_train and self.epoch % 100 == 0:
                self.runner.logger.log_mesh("pointcloud_encoder_input", pointcloud[...,:3].detach(), None, vertex_normals=pointcloud[...,3:6].detach())

        return self.feature.encode(pointcloud)

    def evaluate(self, query_coords, fea=None, **kwargs)->Dict[str, torch.Tensor]:
        is_train = kwargs.get("istrain", self.training)
        detach = kwargs.get('detach', True)
        self.epoch = kwargs.get("epoch", self.epoch)
        force_run = kwargs.get("force_run", False)
        batch_size = query_coords.shape[0]
        outputs = {}

        if kwargs.get("compute_gt", False):
            # compute groundtruth SDF
            try:
                gt_sdf = self.runner.active_task.data.get_sdf(query_coords.view(-1, 3)).view(batch_size, -1)
            except Exception:
                gt_sdf = query_coords.new_ones(query_coords.shape[:-1])*-1

            outputs['gt'] = gt_sdf

        ###### base sdf and normal for query points #######
        # compute gradient. Not necessary to make graph, unless we want to use autograd to compute gradient of sdf wrt p
        # the entire way. Currently we opt for using finitely difference
        with torch.autograd.enable_grad():
            base = self.base({"coords":query_coords, "detach":detach}) # base evalutation
            value = base["sdf"]
            input_points_t = base["detached"]
            grad = base.get("grad", None)
            if grad is None:
                grad = torch.autograd.grad(value, [input_points_t], grad_outputs=torch.ones_like(value), retain_graph=True,
                                        create_graph=True)[0]

        normal = F.normalize(grad,p=2,dim=-1)
        if self.close_surface_activation:
            activation = 1.0 / (1 + (value.detach()/self.activation_threshold)**4)
        else:
            activation = torch.ones_like(value).detach()

        if not force_run and is_train and not self.residual.requires_grad:
            outputs.update({"sdf":torch.zeros_like(value).detach(), "residual":torch.zeros_like(value).detach(),
                    "detached":input_points_t, "activation": activation,
                    "base":value, "base_normal": grad,
                    "prediction":torch.zeros_like(input_points_t), "encoded": fea})
            return outputs

        ###### displacement #######
        if self.approximate_gradient or self.detach_query:
            query_features = self.feature.query_feature(fea, query_coords)
        else:
            query_features = self.feature.query_feature(fea, input_points_t)

        if is_train:
            self.factor.fill_(min(self.epoch*self.offset_base, self.offset_max))
            self.runner.logger.log_scalar("factor", self.factor)

        if self.feature_only:
            displacement = self.residual({"coords": query_features, "detach": False, "epoch": self.epoch})['sdf']
        else:
            if self.detach_normal:
                normal.detach_()

            if self.base_normal_in_c:
                query_features = torch.cat([query_features, normal], dim=-1)
            else:
                pass

            # query_features.add_(torch.randn_like(query_features)*0.05)
            dis_coords = []
            for mode in self.query_modes:
                if mode == "BSDF":
                    _value = value.view(batch_size, -1, 1).clone()
                    if self.scale_base_value:
                        _value = torch.tanh(0.8/self.activation_threshold*_value)
                    dis_coords.append(_value)
                elif mode == "BN":
                    dis_coords.append(normal.clone())
                elif mode == "LC":
                    local_points = self.feature.query_local_coordinates(input_points_t)
                    dis_coords.append(local_points.view(batch_size, -1, 3))

            dis_coords = torch.cat(dis_coords, dim=-1)
            displacement = self.residual({"coords": dis_coords, "x": query_features, "detach": False, "epoch": self.epoch})["sdf"]

        # limit displacement to -1~1
        if self.use_tanh:
            displacement = torch.tanh(displacement)

        displacement = self.factor*activation*displacement
        prediction = input_points_t + displacement*normal.detach()
        result = self.base({"coords":prediction,"detach":False})["sdf"]

        outputs.update({"sdf":result, "residual":displacement, "detached":input_points_t, "activation": activation,
                   "base":value, "base_normal": grad, "prediction":prediction, "encoded": fea})


        return outputs

    def forward(self, args):
        """
        Args:
            coords: (B, N, 3) query coordinates, including
            sdf_out: (B, N, 1) -1/0 for offsurface/onsurface points
            pointcloud: (B, P, 3) (perturbed) input point cloud representing the base shape
        """
        coords = args['coords']
        self.epoch = args.get("epoch", self.epoch)

        ###### query point feature from the feature net #######
        features = self.encode(args)
        outputs = self.evaluate(coords, features, **args)

        return outputs

    def save(self, path):
        torch.save(self, path)


