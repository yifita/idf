from typing import Dict
from evaluator.evaluator import Evaluator
import torch
import numpy as np
import matplotlib.pyplot as plt

from network.siren import gradient
from helper import AllOf
from logger.utils import get_logger

class Normal(Evaluator):

    def __init__(self, config):
        self.attributes = ["sdf"]
        #self.input = "detached"
        self.frequency =  10
        self.resolution = 200
        self.qresolution = 40
        self.axis = 0
        self.offset = 0
        self.compute_gt :bool = True
        self.bbox_size = 2.0
        super().__init__(config)

    def evaluate_level(self, data):
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size
        xlist = np.linspace(-0.5, 0.5, self.resolution)*bbox_size
        xqlist = np.linspace(0,self.resolution-1,self.qresolution,dtype=int)
        ylist = np.linspace(-0.5, 0.5, self.resolution)*bbox_size
        yqlist = np.linspace(0,self.resolution-1,self.qresolution,dtype=int)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.ones_like(X)*self.offset
        filter = [1,2]
        if(self.axis == 0):
            coords = np.stack([Z, Y, X], 2)
            filter = [2,1]
        elif (self.axis == 1):
            coords = np.stack([X, Z, Y], 2)
            filter = [0,2]
        else:
            coords = np.stack([X, Y, Z], 2)
            filter = [0,1]

        coords = torch.from_numpy(coords).float().cuda()
        coords = coords.reshape([1, -1,3])
        inp = {"detach" : True,"coords" : coords,"compute_gt" : self.compute_gt }
        model_output = self.runner.network(inp)


        for attr in self.attributes:
            if(not isinstance(attr,AllOf)):
                pred_sdf = model_output[attr]
                coords = model_output["detached"]
                gradientV = gradient(pred_sdf, coords, graph=False).reshape([-1,3]).cpu().detach().numpy()
                pred_sdf = pred_sdf.flatten()
                name = f"{self.name}_{attr}"
            else:
                gradientV = model_output[attr[1]].reshape([-1,3]).cpu().detach().numpy()
                pred_sdf =  model_output[attr[0]].reshape([-1])
                name = f"{self.name}_{attr[0]}"

            gradientV3 = gradientV
            gradientV3 /= np.linalg.norm(gradientV3, axis=-1,keepdims=True)
            #gradientV3 = gradientV3[:,self.axis]
            gradientV = gradientV[...,filter]
            gradientV /= np.linalg.norm(gradientV, axis=-1,keepdims=True)
            fig, ax = plt.subplots(1,1,dpi=200)
            ax.imshow(gradientV3.reshape([self.resolution,self.resolution,3]),origin='lower',extent=np.array([-.5,.5,-.5,.5])*bbox_size,alpha=0.5)
            ax.contour(X,Y,pred_sdf.detach().cpu().numpy().reshape([self.resolution,self.resolution]),[0])
            gradientV = gradientV.reshape([self.resolution,self.resolution,2])[xqlist,:,:][:,yqlist,:]
            ax.quiver(X[xqlist,:][:,yqlist],Y[yqlist,:][:,xqlist],gradientV[:,:,0],gradientV[:,:,1], minshaft =2)
            self.runner.logger.log_figure(name, fig)

    def epoch_hook(self, epoch : int, data: Dict=None):
        if(epoch % self.frequency == 0):
            self.evaluate_level(data)