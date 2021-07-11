import json
from typing import Dict, Union
from .logger import Logger
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import os
import shutil

class Tensorboard(Logger):

    def __init__(self, config):
        self.writer : SummaryWriter = None
        super().__init__(config)

    def on_task(self, name):
        super().on_task(name)
        if(self.writer is not None):
            self.writer.flush()
        if self._active:
            # if(os.path.exists(self.log_folder)):
            #     shutil.rmtree(self.log_folder)
            self.writer = SummaryWriter(self.log_folder)
        else:
            self.writer = None

    def _log_config(self, config: str):
        import json
        self.config = config
        self.writer.add_text("CONFIG", self.config)

    def _log_graph(self, net, inputs):
        self.writer.add_graph(net, inputs)

    def _log_text(self, name:str, value:str):
       if(self.writer is None):
            return
       self.writer.add_text(name, value,  global_step = self.step)

    def _log_scalar(self, name:str, value : Union[Tensor, Dict]):
        if isinstance(value, Dict):
            self.writer.add_scalars(name, value, self.step)
        else:
            self.writer.add_scalar(name, value, self.step)

    def _log_image(self, name:str, image : Tensor):
        self.writer.add_image(name,image, self.step)

    def _log_mesh(self, name:str, vertices : Tensor, faces : Tensor, colors : Tensor, vertex_normals:Tensor = None):
        self.writer.add_mesh(name, vertices, colors=colors, faces=faces, global_step = self.step)

    def _log_hist(self, name:str, values : Tensor):
        try:
            self.writer.add_histogram(name, values, global_step = self.step)
        except:
            pass

    def _log_figure(self, name:str, figure):
        self.writer.add_figure(name, figure, self.step)

