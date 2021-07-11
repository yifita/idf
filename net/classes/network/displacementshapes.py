from typing import Dict
from network.network import Network
from network.siren import gradient
import torch
import torch.nn.functional as F
from .sphere import Sphere
from .ellipsoid import Ellipsoid

class DisplacementShapes(Network):

    def __init__(self, config):
        self.feature : Network = None
        self.base : Network = None
        self.residual : Network = None
        self.freeze_base : bool = False
        self.offset_base : float = 0.02
        self.offset_max : float = 0.1
        self.pointcloud_sigma: float = 0.0
        self.pointcloud_size: int = 3000
        self.close_surface_activation: bool = False
        self.activation_threshold: float = 0.2
        self.approximate_gradient: bool = False
        self.scale_base_value: bool = False  # scale base value before inputting to residual net
        self.use_base_normal: bool = False  # input base normal to residual net
        super().__init__(config)

    def _initialize(self):
        self.epoch = 0
        if self.freeze_base and self.base is not None:
             self.base.requires_grad_(False)

        self.register_buffer('factor', torch.tensor(self.offset_max))

    def encode(self, args:Dict):
        """ Given pointclouds encode feature """
        pointcloud_size = self.pointcloud_size
        try:
            pointcloud_size = self.runner.active_task.pointcloud_size
        except Exception:
            pass

        if isinstance(self.base, (Sphere, Ellipsoid)):
            pointcloud, _ = self.base.generate_point_cloud(pointcloud_size, data=None)
            pointcloud = pointcloud.cuda().view(1, -1, 3)
        else:
            pointcloud = args['pointcloud']

        if self.pointcloud_sigma > 0.0:
            pointcloud += self.pointcloud_sigma * torch.randn_like(pointcloud)

        if args is not None:
            is_train = args.get("istrain", self.training)
            self.epoch = args.get("epoch", self.epoch)
            if is_train and self.epoch % 100 == 0:
                self.runner.logger.log_mesh("pointcloud_encoder_input", pointcloud[...,:3].detach(), None, vertex_normals=pointcloud[...,3:6].detach())
        return self.feature.encode(pointcloud)

    def evaluate(self, query_coords, fea=None, **kwargs):
        is_train = kwargs.get("istrain", self.training)
        self.epoch = kwargs.get('epoch', self.epoch)
        detach = kwargs.get("detach", True)
        batch_size = query_coords.shape[0]

        outputs = {}
        if kwargs.get("compute_gt", False):
            # compute groundtruth SDF
            try:
                gt_sdf = self.runner.active_task.data.get_sdf(query_coords.view(-1, 3)).view(batch_size, -1)
            except Exception:
                gt_sdf = kwargs.get("sdf_out", None) or kwargs.get("occupancy_out", None) or query_coords.new_ones(query_coords.shape[:-1])*-1

            outputs['gt'] = gt_sdf

        if (not detach and not query_coords.requires_grad):
            detach = True

        with torch.autograd.enable_grad():
            global_fea = self.feature.query_global_feature(fea)
            base = self.base({"coords":query_coords, "x": global_fea, "detach":detach})
            value = base["sdf"][...,0]
            input_points_t = base["detached"]
            grad = base.get("grad", None)
            if grad is None:
                grad = torch.autograd.grad(value, [input_points_t], grad_outputs=torch.ones_like(value),retain_graph=True,
                                        create_graph=True)[0]

        normal = F.normalize(grad, p=2, dim=-1)

        if self.close_surface_activation:
            activation = 1.0 / (1 + (value.detach()/self.activation_threshold)**4)
            self.runner.logger.log_hist("activation", activation)
        else:
            activation = torch.ones_like(value).detach()

        outputs.update({"sdf":torch.zeros_like(value).detach(), "residual":torch.zeros_like(value).detach(),
                "detached":input_points_t, "activation": activation,
                "base":value, "base_normal": grad,
                "prediction":torch.zeros_like(input_points_t), "encoded": fea})


        if is_train and not self.residual.requires_grad:
            return outputs

        ###### displacement #######
        query_features = self.feature.query_feature(fea, input_points_t)
        # query_features = base['sdf'][...,1:]

        if self.use_base_normal:
            # dis_coords = torch.cat([normal, value.view(batch_size, -1, 1)], dim=-1)
            dis_coords = value.view(batch_size, -1, 1).clone()

            if self.scale_base_value:
                dis_coords = torch.tanh(5*dis_coords)
            if self.use_base_normal:
                query_features = torch.cat([query_features, normal], dim=-1)

            residual = self.residual({"coords": dis_coords, "x": query_features, "detach": False, "epoch": self.epoch})["sdf"]
        else:
            dis_coords = input_points_t.clone()
            residual = self.residual({"coords": dis_coords, "x": query_features, "detach": False, "epoch": self.epoch})["sdf"]


        if is_train:
            self.factor.fill_(min(self.epoch*self.offset_base, self.offset_max))
            self.runner.logger.log_scalar("factor", self.factor)

        displacement = self.factor*residual*activation
        prediction = input_points_t + displacement*normal.detach()

        result = self.base({"coords":prediction, "x": global_fea, "detach":False})["sdf"]


        outputs.update({"sdf":result, "residual":displacement, "detached":input_points_t, "activation": activation,
                "base":value, "base_normal":grad, "prediction":prediction, "encoded": fea})

        if is_train and self.training and (self.epoch % 5 == 0):
            self.runner.logger.log_hist("query_feature", query_features.detach())
            self.runner.logger.log_hist("base_sdf", value.detach())

        return outputs

    def forward(self, args):
        coords = args['coords']
        self.epoch = args.get("epoch", self.epoch)

        ###### query point feature from the feature net #######
        features = self.encode(args)
        outputs = self.evaluate(coords, features, **args)

        return outputs


    def save(self, path):
        torch.save(self, path)
        torch.save(self.base, path[:-4] + "_base" + path[-4:])