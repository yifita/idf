"""
Convolutional Feature Embedding
"""
from typing import Callable, Dict, List, Tuple, Union
import torch
from .custom import grid_sample_2d, grid_sample_3d
from torch import nn
from torch_scatter import scatter_mean, scatter_max
from network.network import Network
from network.ocn.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate
from network.ocn.layers import ResnetBlockFC
from network.ocn.unet import UNet
from network.ocn.unet3d import UNet3D
import numpy as np


def get_knn(vs: torch.Tensor, k: int, batch_idx: torch.Tensor) -> torch.Tensor:
    mat_square = torch.matmul(vs, vs.transpose(2, 1))
    diag = torch.diagonal(mat_square, dim1=1, dim2=2)
    diag = diag.unsqueeze(2).expand(mat_square.shape)
    dist_mat = (diag + diag.transpose(2, 1) - 2 * mat_square)
    _, index = dist_mat.topk(k + 1, dim=2, largest=False, sorted=True)
    index = index[:, :, 1:].view(-1, k) + batch_idx[:, None] * vs.shape[1]
    return index.flatten()


def extract_angles(vs: torch.Tensor, distance_k: torch.Tensor, vs_k: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
    proj = torch.einsum('nd,nkd->nk', vs, vs_k)
    cos_angles = torch.clamp(proj / distance_k, -1., 1.)
    proj = vs_k - vs[:, None, :] * proj[:, :, None]
    # moving same axis points
    ma = torch.abs(proj).sum(2) == 0
    num_points_to_replace = ma.sum().item()
    if num_points_to_replace:
        proj[ma] = torch.rand(num_points_to_replace,
                              vs.shape[1], device=ma.device)
    proj = proj / torch.norm(proj, p=2, dim=2)[:, :, None]
    angles = torch.acos(cos_angles)
    return angles, proj


def min_angles(dirs: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ref = dirs[:, 0]
    all_cos = torch.einsum('nd,nkd->nk', ref, dirs)
    all_sin = torch.cross(ref.unsqueeze(
        1).expand(-1, dirs.shape[1], -1), dirs, dim=2)
    all_sin = torch.einsum('nd,nkd->nk', up, all_sin)
    all_angles = torch.atan2(all_sin, all_cos)
    all_angles[:, 0] = 0
    all_angles[all_angles < 0] = all_angles[all_angles < 0] + 2 * np.pi
    all_angles, inds = all_angles.sort(dim=1)
    inds = torch.argsort(inds, dim=1)
    all_angles_0 = 2 * np.pi - all_angles[:, -1]
    all_angles[:, 1:] = all_angles[:, 1:] - all_angles[:, :-1]
    all_angles[:, 0] = all_angles_0
    all_angles = torch.gather(all_angles, 1, inds)
    return all_angles


def extract_rotation_invariant_features(k: int) -> Tuple[Callable[[Union[torch.Tensor, np.array]], torch.Tensor], int]:

    batch_idx = None
    num_features = k * 3 + 1

    def get_input(xyz: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        nonlocal batch_idx
        if batch_idx is None or len(batch_idx) != xyz.shape[0] * xyz.shape[1]:
            batch_idx, _ = torch.meshgrid(
                [torch.arange(xyz.shape[0]), torch.arange(xyz.shape[1])])
            batch_idx = batch_idx.flatten().to(xyz.device)
        return xyz.view(-1, 3), batch_idx

    def extract(base_vs: Union[torch.Tensor, np.array]):
        nonlocal num_features
        with torch.no_grad():
            if type(base_vs) is np.array:
                base_vs = torch.Tensor(base_vs)
            batch_size, num_pts = base_vs.shape[:2]
            vs, batch_idx = get_input(base_vs)
            knn = get_knn(base_vs, k, batch_idx)
            vs_k = vs[knn].view(-1, k, vs.shape[1])
            distance = torch.norm(vs, p=2, dim=1)
            vs_unit = vs / distance[:, None]
            distance_k = distance[knn].view(-1, k)
            angles, proj_unit = extract_angles(vs_unit, distance_k, vs_k)
            proj_min_angle = min_angles(proj_unit, vs_unit)
            fe = torch.cat([distance.unsqueeze(1), distance_k,
                            angles, proj_min_angle], dim=1)
        return fe.view(batch_size, num_pts, num_features)

    return extract, num_features


class ConvolutionalFeature(Network):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Attributes:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        reso_plane (int): defined resolution for plane feature
        reso_grid (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, config):
        # defaults from shapenet 3plane
        self.c_dim: int = 64
        self.dim: int = 3
        self.hidden_dim: int = 32
        self.scatter_type: str = 'max'
        self.unet: bool = False
        self.unet_kwargs: dict = dict(
            depth=4, merge_mode='concat', start_filts=32)
        self.unet3d: bool = False
        self.unet3d_kwargs: dict = None
        self.reso_plane: int = 64
        self.reso_grid: int = 32
        self.subpixel_upsampling: int = 1  # ratio for subpixel upsampling
        self.plane_type: List[str] = ['xz', 'xy', 'yz']
        self.n_blocks: int = 5
        self.sample_mode: str = "bilinear"
        self.input_normals: bool = False
        self.bbox_size: float = 2.0
        self.clusternet_feature: bool = False
        super().__init__(config)

    def _initialize(self):
        # shortcuts
        hidden_dim = self.hidden_dim
        dim = self.dim
        n_blocks = self.n_blocks
        c_dim = self.c_dim
        unet_kwargs = self.unet_kwargs
        unet = self.unet
        unet3d = self.unet3d

        if self.clusternet_feature:
            self.extractor, dim = extract_rotation_invariant_features(8)
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim * (self.subpixel_upsampling**2),
                             in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            unet3d_kwargs = self.unet3d_kwargs.copy()
            unet3d_kwargs["out_channels"] *= (self.subpixel_upsampling ** 3)
            self.unet3d = UNet3D(**self.unet3d_kwargs)
        else:
            self.unet3d = None

        if self.subpixel_upsampling > 1:
            self.ps = torch.nn.PixelShuffle(self.subpixel_upsampling)

        if self.scatter_type == 'max':
            self.scatter = scatter_max
        elif self.scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.last_epoch = -1

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        # normalize to the range of (0, 1)
        xy = normalize_coordinate(
            p.clone(), plane=plane, bbox_size=self.bbox_size)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)  # B x 512 x torch.Tensor
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        # sparce matrix (B x 512 x reso x reso)
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        if self.subpixel_upsampling > 1:
            # pixelshuffle upsampling
            fea_plane = self.ps(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), bbox_size=self.bbox_size)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        # sparce matrix (B x 512 x reso x reso)
        fea_grid = fea_grid.reshape(
            p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        if self.subpixel_upsampling > 1:
            fea_grid = depthToVolume(fea_grid, self.subpixel_upsampling)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1),
                                   index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1),
                                   index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def query_feature(self, fea: Dict[str, torch.Tensor], query_coords: torch.Tensor) -> torch.Tensor:
        self.bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)

        # originally inside the decoder
        if self.c_dim != 0:
            plane_type = list(fea.keys())
            # merge different planes with sum
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(query_coords, fea['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(query_coords,
                                               fea['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(query_coords,
                                               fea['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(query_coords,
                                               fea['yz'], plane='yz')
            c = c.transpose(1, 2)

        return c

    def query_global_feature(self, fea: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Global max and avg pooling
        Returns:
            (B, P, 2*self.c_dim)
        """
        if self.c_dim != 0:
            plane_type = list(fea.keys())
            # merge different planes with sum
            c = 0
            if 'grid' in plane_type:
                batch_size = fea['grid'].shape[0]
                c_max = fea['grid'].view(
                    batch_size, self.c_dim, -1).max(dim=-1)[0]
                c_avg = fea['grid'].view(
                    batch_size, self.c_dim, -1).mean(dim=-1)
                c += torch.cat([c_max, c_avg], dim=-1)
            if 'xz' in plane_type:
                batch_size = fea['xz'].shape[0]
                c_max = fea['xz'].view(
                    batch_size, self.c_dim, -1).max(dim=-1)[0]
                c_avg = fea['xz'].view(batch_size, self.c_dim, -1).mean(dim=-1)
                c += torch.cat([c_max, c_avg], dim=-1)
            if 'xy' in plane_type:
                batch_size = fea['xy'].shape[0]
                c_max = fea['xy'].view(
                    batch_size, self.c_dim, -1).max(dim=-1)[0]
                c_avg = fea['xy'].view(batch_size, self.c_dim, -1).mean(dim=-1)
                c += torch.cat([c_max, c_avg], dim=-1)
            if 'yz' in plane_type:
                batch_size = fea['yz'].shape[0]
                c_max = fea['yz'].view(
                    batch_size, self.c_dim, -1).max(dim=-1)[0]
                c_avg = fea['yz'].view(batch_size, self.c_dim, -1).mean(dim=-1)
                c += torch.cat([c_max, c_avg], dim=-1)

        return c

    def sample_plane_feature(self, p, c, plane='xz'):
        # normalize to the range of (0, 1)
        xy = normalize_coordinate(
            p.clone(), plane=plane, bbox_size=self.bbox_size)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        if c.shape[0] == 1 and vgrid.shape[0] > 0:
            c = c.expand(vgrid.shape[0], -1, -1, -1)
        c = grid_sample_2d(c, vgrid).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        # normalize to the range of (0, 1)
        p_nor = normalize_3d_coordinate(
            p.clone(), bbox_size=self.bbox_size)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        if c.shape[0] == 1 and vgrid.shape[0] > 0:
            c = c.expand(vgrid.shape[0], -1, -1, -1, -1)
        c = grid_sample_3d(c, vgrid).squeeze(-1).squeeze(-1)
        return c

    def encode(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        input_coords = input[..., :3]
        if self.input_normals:
            input_feats = input[..., 3:(3 + self.dim)]
        elif self.clusternet_feature:
            input_feats = self.extractor(input_coords)
        else:
            input_feats = input_coords.clone()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(input_coords.clone(
            ), plane='xz', bbox_size=self.bbox_size)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(input_coords.clone(
            ), plane='xy', bbox_size=self.bbox_size)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(input_coords.clone(
            ), plane='yz', bbox_size=self.bbox_size)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(
                input_coords.clone(), bbox_size=self.bbox_size)
            index['grid'] = coordinate2index(
                coord['grid'], self.reso_grid, coord_type='3d')

        net = self.fc_pos(input_feats)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(input_coords, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(
                input_coords, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(
                input_coords, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(
                input_coords, c, plane='yz')

        return fea

    def query_local_coordinates(self, input_coords):
        self.bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)

        if "grid " in self.plane_type and len(self.plane_type) > 1:
            # make sure plane resolution is the same as grid resolution
            assert(self.reso_grid ==
                   self.reso_plane), "Local coordinates mapping only supports equal grid_reso and plane_reso"

        # assuming input in between -0.5 to 0.5
        p_nor = normalize_3d_coordinate(
            input_coords.clone(), bbox_size=self.bbox_size)
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        if 'grid' in self.plane_type:
            cell_size = 2.0 / self.reso_plane
        else:
            cell_size = 2.0 / self.reso_grid

        return torch.remainder(vgrid, cell_size) / cell_size

    def forward(self, args) -> torch.Tensor:
        """
        Args:
            input_coords: (N, P, 3) input point positions (can use relative ones as well)
            query_coords (N, Q, 3) query point position
        Returns:
            query_fea: (N, P, c_dim)
        """
        self.bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)

        input_coords = args['pointclouds']
        query_coords = args['coords']

        assert(query_coords.size(-1) == 3)

        fea = self.encode(input_coords)
        query_feat = self.query_feature(fea, query_coords)

        return query_feat

    def save(self, path):
        torch.save(self, path)


def depthToVolume(x, upsample_ratio):
    N, C, D, H, W = x.shape
    x = x.view(N, upsample_ratio, upsample_ratio, upsample_ratio, C //
               (upsample_ratio ** 3), D, H, W)  # (N, bs, bs, bs, C//bs^3, D, H, W)
    # (N, C//bs^2, D, bs H, bs, W, bs)
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    x = x.view(N, C // (upsample_ratio ** 3), D * upsample_ratio, H *
               upsample_ratio, W * upsample_ratio)  # (N, C//bs^2, H * bs, W * bs)
    return x


def volumeToDepth(x, upsample_ratio):
    N, C, D, H, W = x.shape
    x = x.view(N, C, D // upsample_ratio, upsample_ratio, H // upsample_ratio, upsample_ratio,
               W // upsample_ratio, upsample_ratio)  # (N, C, D//bs, bs, H//bs, bs, W//bs, bs)
    # (N, bs, bs, bs, C, D//bs, H//bs, W//bs)
    x = x.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()
    x = x.view(N, C * (upsample_ratio ** 3), H // upsample_ratio,
               W // upsample_ratio)  # (N, C*bs^2, H//bs, W//bs)
    return x
