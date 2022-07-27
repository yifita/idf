import importlib
import logging
import os
from config import ConfigObject

class Task(ConfigObject):


    def __init__(self, config):
        self.name : str = "Task"
        self.overwrite: bool = False
        super().__init__(config)

    @classmethod
    def get_class_for(self, name:str, classname: str):
        basename = name
        basename = basename + '.' + classname.lower()
        return getattr(importlib.import_module(basename),classname)

    def __call__(self):
        """ Set logger directories """
        self.folder = os.path.join(self.runner.folder, self.name)
        # check whether directory exist
        if not self.overwrite:
            count = 0
            while(os.path.exists(self.folder)):
                count +=1
                self.folder = os.path.join(self.runner.folder, f"{self.name}_{count}")

        os.makedirs(self.folder, exist_ok=True)
        self.name = os.path.basename(self.folder)
        # Set child logger
        self.runner.py_logger = logging.getLogger(f'{self.runner.name}.{self.name}')
        # close filehandle if exist, create new filehandle in the task directory
        for h in self.runner.py_logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
                self.runner.py_logger.removeHandler(h)
        ch = logging.FileHandler(os.path.join(self.folder, 'task.log'))
        ch.setLevel(logging.INFO)
        self.runner.py_logger.addHandler(ch)
        self.runner.logger.on_task(self.name)