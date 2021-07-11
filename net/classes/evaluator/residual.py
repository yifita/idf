from typing import Dict

from matplotlib.colors import TwoSlopeNorm
from evaluator.evaluator import Evaluator
import torch
import numpy as np
import matplotlib.pyplot as plt

class Residual(Evaluator):

    def __init__(self, config):
        self.attributes = ["sdf"]
        self.frequency =  10
        self.resolution = 200
        self.axis = 0
        self.offset = 0
        self.batch_size = 125000
        self.bbox_size = 2.0
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
            coordsL = torch.Tensor(coords[start:end,:]).cuda()
            eval_results = self.evaluate_network(coordsL.unsqueeze(0), fea=fea, **data)
            for attr in self.attributes :
                if(i==0):
                    results[attr] = np.zeros([self.resolution**2])
                result = eval_results[attr]
                results[attr][start:end] = result.cpu().detach().numpy().flatten()
                del result
            del eval_results
            del coordsL

        for attr in self.attributes :
            result = results[attr]
            name = f"{self.name}_{attr}"
            sdf = result.reshape(X.shape)
            fig = plt.figure()
            vmax = np.max(sdf)
            vmin = np.min(sdf)
            if(vmax <= 0):
                vmax = 1e-3
            if(vmin >= 0):
                vmin = -1e-3

            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            plt.imshow(sdf.reshape([self.resolution,self.resolution]), origin='lower',norm=norm,
                       extent=np.array([-.5,.5,-.5,.5])*bbox_size,cmap="seismic")
            plt.colorbar(ticks=[vmin, 0, vmax])
            plt.contour(X,Y,sdf.reshape([self.resolution,self.resolution]),[0])
            if 'base' in self.attributes:
                plt.contour(X,Y, results['base'].reshape([self.resolution, self.resolution]),[0], linestyles="dashed")
            if 'gt' in self.attributes:
                plt.contour(X,Y, results['gt'].reshape([self.resolution, self.resolution]),[0], colors='yellow', linestyles='dotted')

            self.runner.logger.log_figure(name, fig)


    def epoch_hook(self, epoch : int, data: Dict):
        if epoch % self.frequency == 0:
            self.runner.py_logger.info(f"Generating residual plot {self.name} with resolution {self.resolution}")
            self.evaluate_level(data)