from logger.utils import get_logger
from task.task import Task
import torch
from torch.utils.data import DataLoader

class Save(Task):

    def __init__(self, config):
        super().__init__(config)
        if(not hasattr(self,"path")):
            self.path = f"runs/{self.runner.name}/{self.name}.pt"

    def __call__(self):
        self.runner.py_logger.info(f"Saving model to {self.path}\n")
        self.runner.network.save(self.path)
        self.runner.py_logger.info(f"model saved\n")
