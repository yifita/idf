from evaluator.evaluator import Evaluator

class Scalar(Evaluator):

    def __init__(self, config):
        self.attributes = []
        super().__init__(config)

    def train_hook(self, model_output:dict, loss_output:dict):
        for attr in self.attributes:
            res = model_output.get(attr,None)
            if(res is None):
                res = loss_output.get(attr, None)
            if(res is not None):
                self.runner.logger.log_scalar(f"{self.name}_{attr}", res.mean())
