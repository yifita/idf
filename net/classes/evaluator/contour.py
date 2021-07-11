from typing import Dict
from evaluator.evaluator import Evaluator
from skimage import measure
from skimage.draw import ellipsoid
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


cmap = plt.get_cmap('PuOr')
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize and forcing 0 to be part of the colorbar!
sdf_bounds = np.arange(-1, 1, 0.1)
norm = BoundaryNorm(sdf_bounds, cmap.N)

class Contour(Evaluator):

    def __init__(self, config):
        self.attributes = ["sdf"]
        self.frequency =  10
        self.resolution = 200
        self.axis = 0
        self.offset = 0
        self.batch_size = 125000
        self.bbox_size :float = 2.0
        super().__init__(config)

    def evaluate_level(self, data):
        data.pop('coords', None)
        fea = self.encode_network(data)

        torch.cuda.empty_cache()
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size
        xlist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        ylist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.ones_like(X)*self.offset
        if(self.axis == 0):
            coords = np.stack([Z, Y, X], 2)
        elif (self.axis == 1):
            coords = np.stack([X, Z, Y], 2)
        else:
            coords = np.stack([X, Y, Z], 2)

        coords = coords.reshape([-1,3])
        num_batches = int((self.batch_size -1 + self.resolution**2)/self.batch_size)
        results = {}
        for i in range(num_batches):
            start = i*self.batch_size
            end = min(self.resolution**2-1,start+self.batch_size)
            l_coords = torch.Tensor(coords[start:end,:]).cuda()
            eval_results =  self.evaluate_network(l_coords.unsqueeze(0), fea=fea, **data)

            for attr in self.attributes:
                if i==0:
                    results[attr] = np.zeros([self.resolution**2])
                result = eval_results[attr]
                results[attr][start:end] = result.cpu().detach().numpy().flatten()
                del result
            del eval_results
            del l_coords

        for attr in self.attributes :
            result = results[attr]
            name = f"{self.name}_{attr}"
            sdf = result.reshape(X.shape)
            fig = plt.figure()
            plt.contour(X, Y, sdf, levels=sdf_bounds)
            plt.contourf(X, Y, sdf, levels=sdf_bounds, norm=norm, cmap=cmap)
            if 'gt' in self.attributes:
                plt.contour(X,Y, results['gt'].reshape([self.resolution, self.resolution]),[0], colors='yellow', linestyles='dotted')
            plt.colorbar()
            self.runner.logger.log_figure(name, fig)

    def epoch_hook(self, epoch : int, data:Dict=None):
        if(epoch % self.frequency == 0):
            self.runner.py_logger.info(f"Generating contour plot {self.name} with resolution {self.resolution}")
            self.evaluate_level(data)