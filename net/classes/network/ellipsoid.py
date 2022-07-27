from pytorch3d.ops.knn import knn_points
import torch
import torch.nn.functional as F
from  network.network import Network
import numpy as np
from helper import AllOf
from scipy.interpolate import interp1d, interp2d

class Ellipsoid(Network):

    def __init__(self, config):
        self.axis  = AllOf([[1.0, 1.0, 1.0]])
        super().__init__(config)

    def _initialize(self):
        axis = torch.tensor(self.axis.elements, dtype=torch.float32)
        del self.axis
        self.register_buffer('axis', axis)

    def encode(self, *args, **kwargs):
        pass

    def evaluate(self, query_coords, fea=None, **kwargs):
        kwargs.update({'coords': query_coords})
        return self.forward(kwargs)

    def generate_point_cloud(self, n_points:int, data=None):
        # project radially to sphere, non-uniform
        def ellipsoid(t, u, a=self.axis[0].item(), b=self.axis[1].item(), c=self.axis[2].item()):
            return np.array([a*np.sin(u)*np.cos(t), b*np.sin(u)*np.sin(t), c*np.cos(u)])

        domain_t = [0, 2*np.pi]
        domain_u = [0, np.pi]

        # Get random points
        x, _, _, _, _ = r_surface(n_points, ellipsoid, *domain_t, *domain_u, 100, 100)
        x = torch.tensor(x.astype(np.float32)).T.contiguous()
        n = x / (self.axis.cpu()**2)
        n = F.normalize(n, dim=-1)
        return x, n


    def forward(self, args):
        detach = args.get("detach",True)
        input_points = args.get("coords",None)
        batch_size = input_points.shape[0]

        if detach:
            input_points = input_points.clone().detach().requires_grad_(True)

        if not hasattr(self, '_points'):
            self._points, self._normals = self.generate_point_cloud(1000000)
            self._points = self._points.to(device=input_points.device).view(1, -1, 3)
            self._normals = self._normals.to(device=input_points.device).view(1, -1, 3)
            self._normals = F.normalize(self._normals,dim=-1)
            # self.runner.logger.log_mesh('ellipse_dense', self._points, None, vertex_normals=self._normals)

        knn_result = knn_points(input_points, self._points.expand(batch_size, -1, -1), K=1)
        sign = torch.sum((input_points / self.axis)**2, dim=-1) > 1
        sign = sign.to(dtype=input_points.dtype)
        sign[sign==0.0] = -1.0
        result = knn_result.dists[:,:,0].sqrt() * sign

        normals = 2*(input_points/self.axis)
        return {"sdf":result, "detached":input_points, "grad": normals}

    def save(self, path):
        torch.save(self, path)


def surface_cumulator(t, u, coords):
    """
    Parameters
    ----------
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    u : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t, u pair.
    Returns
    -------
    t, u : arrays
        As above.
    cum_S_t : array
        Cumulative surface area on [t[0], t], all u.
    cum_S_u : array
        Cumulative surface area on all t, [u[0], u].
    Evaluates the cumulative surface area at each coordinate.
    """

    if np.all(t) is None:
        t, _ = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))
    if np.all(u) is None:
        _, u = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))

    assert t.shape == u.shape == coords.shape[1:], \
        "Need same number of parameters as coordinates"
    delta_t_temp = np.diff(coords, axis=2)
    delta_u_temp = np.diff(coords, axis=1)

    # Pad with zeros so that small rand_S can still be interpd
    delta_t = np.zeros(coords.shape)
    delta_u = np.zeros(coords.shape)

    delta_t[:coords.shape[0], :coords.shape[1], 1:coords.shape[2]] = delta_t_temp
    delta_u[:coords.shape[0], 1:coords.shape[1], :coords.shape[2]] = delta_u_temp

    # Area of each parallelogram
    delta_S = np.linalg.norm(np.cross(delta_t, delta_u, 0, 0), axis=2)

    # integrate over t
    S_u = delta_S.sum(axis=1)
    S_t = delta_S.sum(axis=0)

    # if the integral is too small, will rarely sample at this region
    # the first row is the prepended 0
    S_u_temp = S_u[1:]
    S_u_temp[S_u_temp / S_u_temp.max() < 1e-3] = S_u_temp[S_u_temp / S_u_temp.max() >= 1e-3].min()
    S_u[1:] = S_u_temp
    S_t_temp = S_t[1:]
    S_t_temp[S_t_temp / S_t_temp.max() < 1e-3] = S_t_temp[S_t_temp / S_t_temp.max() >= 1e-3].min()
    S_t[1:] = S_t_temp

    cum_S_t = np.cumsum(S_t)
    cum_S_u = np.cumsum(S_u)

    # if cum_S_u == 0, will not interpolate
    return t, u, cum_S_t, cum_S_u


def r_surface_from_data(n, t, u, coords, interp=True, kind='linear'):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    u : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t, u pair.
    interp : boolean
        Whether to generate random function values or not. Set to false if you
        have another way of evaluating the function.
    kind : str
        Interpolation method to be passed to scipy.interpolate.
    Returns
    -------
    (rand_coords : array)
        Random coordinates, if interp=True was passed.
    rand_t : array
        t-values associated with the coordinates.
    rand_u : array
        u_values associated with the coordinates.
    rand_S_t : array
        Cumulative area at each t-value over full range of u.
    rand_S_u : array
        Cumulative area at each u-value over full range of t.
    If the parameterizing function is known, use r_surface() instead, as the
    coordinates will be exactly computed from t instead of interpolating.
    """

    t, u, cum_S_t, cum_S_u = surface_cumulator(t, u, coords)

    # Random values
    rand_S_t = np.random.rand(n) * cum_S_t[-1]
    rand_S_u = np.random.rand(n) * cum_S_u[-1]

    # Find corresponding t-values by interpolation
    rand_t = interp1d(cum_S_t, t[0, :], )(rand_S_t)
    rand_u = interp1d(cum_S_u, u[:, 0])(rand_S_u)

    if interp:
        # Interpolate coordinates, e.g. if func unknown

        rand_coords = np.empty([coords.shape[0], n])

        # One axis at a time, or else scipy throws dim mismatch
        for i in range(coords.shape[0]):
            f = interp2d(t, u, coords[i], kind=kind)

            # One point at time, or else scipy does a meshgrid
            for j in range(n):
                rand_coords[i, j] = f(rand_t[j], rand_u[j])

        return rand_coords, rand_t, rand_u, rand_S_t, rand_S_u

    else:
        return rand_t, rand_u, rand_S_t, rand_S_u


def r_surface(n, func, t0, t1, u0, u1, t_precision=25, u_precision=25):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1, u0, u1 : ints or floats
        Range over which func is evaluated.
    t_precision, u_precision : ints
        Number of t-values at which func is evaluated when computing
        surface area.
    Returns
    -------
    rand_coords : array
        Random coordinates.
    rand_t : array
        t-values associated with the coordinates.
    rand_u : array
        u_values associated with the coordinates.
    rand_S_t : array
        Cumulative area at each t-value over full range of u.
    rand_S_u : array
        Cumulative area at each u-value over full range of t.
    Generates random points distributed uniformly over a parametric surface.
    """

    t, u = np.meshgrid(np.linspace(t0, t1, t_precision),
                       np.linspace(u0, u1, u_precision))
    coords = func(t, u)

    rand_t, rand_u, rand_S_t, rand_S_u = r_surface_from_data(n, t, u, coords, interp=False)
    rand_coords = func(rand_t, rand_u)

    return rand_coords, rand_t, rand_u, rand_S_t, rand_S_u


def surface_area(func, t0, t1, u0, u1, t_precision=25, u_precision=25):
    """
    Parameters
    ----------
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1, u0, u1 : ints or floats
        Range over which func is evaluated.
    t_precision, u_precision : ints
        Number of t- and u-values at which func is evaluated.
    Returns
    -------
    area : float
        Estimate of surface area.
    Convenience function to evaluate total surface area over given range.
    """

    t, u = np.meshgrid(np.linspace(t0, t1, t_precision), np.linspace(u0, u1, u_precision))
    coords = func(t, u)

    area = surface_cumulator(t, u, coords)[3][-1]

    return area