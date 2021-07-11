from data.dataset import CDataset
import numpy as np
import torch 
from scipy.interpolate import interp1d, interp2d
from helper import AllOf


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

    cum_S_t = np.cumsum(delta_S.sum(axis=0))
    cum_S_u = np.cumsum(delta_S.sum(axis=1))

    return t, u, cum_S_t, cum_S_u




def r_surface_from_data(n, t, u, coords, interp=True, kind='linear'):
    t, u, cum_S_t, cum_S_u = surface_cumulator(t, u, coords)

    # Random values
    rand_S_t = np.random.rand(n) * cum_S_t[-1]
    rand_S_u = np.random.rand(n) * cum_S_u[-1]

    # Find corresponding t-values by interpolation
    rand_t = interp1d(cum_S_t, t[0, :])(rand_S_t)
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
    

    t, u = np.meshgrid(np.linspace(t0, t1, t_precision),
                    np.linspace(u0, u1, u_precision))
    coords = func(t, u)

    rand_t, rand_u, rand_S_t, rand_S_u = r_surface_from_data(n, t, u, coords, interp=False)
    rand_coords = func(rand_t, rand_u)

    return rand_coords, rand_t, rand_u, rand_S_t, rand_S_u


class Ellipsoid(CDataset):

    def ellipsoid(self, t, u):
        return np.array([self.axis[0]*np.sin(u)*np.cos(t), self.axis[1]*np.sin(u)*np.sin(t), self.axis[2]*np.cos(u)])

    def __init__(self,config):
        self.radius :int = 0.5
        self.batch_size : int = 100000
        self.num_points : int = 1000000
        self._coord_min = np.array([0,0,0]).reshape(1,-1)
        self._coord_max = np.array([1,1,1]).reshape(1,-1)
        self.factor_off_surface : float = 0.5
        self.axis  = AllOf([[1.0,1.0,1.0]])
        super().__init__(config)

        domain_t = [0, 2*np.pi]
        domain_u = [0, np.pi]
        # Get random points
        x, t, u, St, Su = r_surface(self.num_points, self.ellipsoid, *domain_t, *domain_u, 200, 200)
        x = x.swapaxes(0,1)
        self._coords = x
        axis = np.array([self.axis.elements])**2
        self._normals = self._coords/axis
        # np.savetxt("/tmp/test.xyz", np.concatenate([self._coords,self._normals], axis = 1), delimiter=' ')
        
    def __len__(self):
        return (self.num_points// self.batch_size) + 1
     
    def __getitem__(self, idx):
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples
        # Random coords
        rand_idcs = np.random.choice(point_cloud_size,
                                     size=on_surface_samples)

        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]

        sdf = torch.zeros((total_samples, 1))  # on-surface = 0

        if(off_surface_samples > 0):
            off_surface_coords = np.random.uniform(-1, 1,
                                                size=(off_surface_samples, 3))
            off_surface_normals = np.ones((off_surface_samples, 3)) * -1

            sdf[on_surface_samples:, :] = -1  # off-surface = -1

            coords = np.concatenate((on_surface_coords, off_surface_coords),
                                    axis=0)
            normals = np.concatenate((on_surface_normals, off_surface_normals),
                                    axis=0)
        else:
            coords = on_surface_coords
            normals = on_surface_normals

        return {
                "coords" : torch.Tensor(coords).float(),
                "normal_out" : torch.Tensor(normals),
                "sdf_out" : torch.Tensor(sdf).float()
                }
     
