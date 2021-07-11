from typing import Dict, List
from config import ConfigObject

class Evaluator(ConfigObject):

    def __init__(self, config):
        self.name : str
        self.evaluate_loss : bool = False
        if( not hasattr(self,"attributes")):
            self.attributes : List = []
        super().__init__(config)

    def train_hook(self, model_output:dict, loss_output: dict):
        pass

    def epoch_hook(self, epoch : int, data: Dict=None):
        pass

    def encode_network(self, data):

        return self.runner.network.encode(data)

    def evaluate_network(self, coords, fea=None, **kwargs):
        """
        Evalaute the network and return all its output for evalyuation
        """
        coords = coords
        if kwargs is not None:
            kwargs['detach'] = True
            kwargs['compute_gt'] = 'gt' in self.attributes

        self.runner.network.eval()
        model_output = self.runner.network.evaluate(coords, fea=fea, **kwargs)

        if(self.evaluate_loss):
            loss_output,_ = self.runner.loss(model_output)
            model_output = {**model_output,**loss_output}

        return model_output

