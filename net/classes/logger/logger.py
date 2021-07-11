import json
from config import ConfigObject
from torch import Tensor
from typing import List,Dict
import os
import re

from runner import Runner


class Logger(ConfigObject):
    def __init__(self, config:Dict[str,object]):
        self.step = 0
        self.include : List[str] = None
        self.exclude : List[str] = []
        self.tasks : List = None
        self._active = False
        super().__init__(config)
        self.name = self.runner.key_value.get("name", "run")

    def set_step(self, step):
        self.step = step

    def filter(self, name:str):
        if(not self._active):
            return False

        if any([re.match(e, name) for e in self.exclude]):
            return False

        if self.include is None:
            return True

        return any([re.match(e, name) for e in self.include])

    def increment(self):
        self.step +=1

    def on_task(self, name:str):
        if(self.tasks is None or any([re.match(t, name) for t in self.tasks])):
            self._active = True
            self.log_folder = os.path.join(self.runner.folder, name, self.__class__.__name__)
            if(not os.path.exists(self.log_folder)):
                os.makedirs(self.log_folder)
        else:
            self._active = False

    def log_graph(self, net, inputs):
        if(not self._active):
            return False
        self._log_graph(net, inputs)


    def log_config(self, config_str: str=None):
        """
        log config object used to create this run
        """
        if config_str is None:
            config_str = json.dumps(Runner.filterdict(self.runner.key_value.copy()), indent=4, skipkeys=True, sort_keys=True)
        self._log_config(config_str)

    def log_text(self, name:str, value:str, *args):
        """
        log_text log text message without step
        :param name: the logging TAG
        :param value: the string to log
        :param args: format string stuff
        """
        if( not self.filter(name)):
            return
        if(len(args) > 0):
            value = value.format(args)

        self._log_text(name, value)
        pass

    def log_scalar(self, name:str, value : Tensor):
        if( not self.filter(name)):
            return
        self._log_scalar(name, value)


    def log_image(self, name:str, image : Tensor):
        if( not self.filter(name)):
            return
        self._log_image(name, image)

    def log_mesh(self, name:str, vertices : Tensor, faces : Tensor, colors: Tensor = None, vertex_normals:Tensor=None):
        if( not self.filter(name)):
            return
        # vertices = self.runner.data.backtransform(vertices)
        self._log_mesh(name, vertices, faces, colors, vertex_normals=vertex_normals)

    def log_hist(self, name:str, values : Tensor):
        if( not self.filter(name)):
            return
        self._log_hist(name, values)

    def log_figure(self, name:str, figure):
        if( not self.filter(name)):
            return
        self._log_figure(name, figure)

    def log_code_base(self):
        self._log_code_base()

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def _log_text(self, name:str, value:str):
        pass

    def _log_config(self, config : str):
        pass

    def _log_scalar(self, name:str, value : Tensor):
        pass

    def _log_figure(self, name:str, figure):
        pass

    def _log_image(self, name:str, image : Tensor):
        pass

    def _log_mesh(self, name:str, vertices : Tensor, faces : Tensor, colors: Tensor, vertex_normals=None):
        pass

    def _log_hist(self, name:str, values : Tensor):
        pass

    def _log_code_base(self):
        pass

    def _log_graph(self, net, inputs):
        pass
