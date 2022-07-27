import os
import torch
from task.task import Task
from torch.utils.data import DataLoader
from checkpoint_io import CheckpointIO
import time

class Eval(Task):

    def __init__(self, config):
        self.pin_memory : bool = True
        self.state_dict_path : str = None
        self.evaluate_loss : bool = False
        super().__init__(config)

    def __call__(self):
        super().__call__()

        if not hasattr(self, 'data') and hasattr(self.runner, 'data'):
            self.data = self.runner.data

        self.runner.network.cuda()
        self.runner.network.eval()

        if self.state_dict_path is not None:
            checkpoint_io = CheckpointIO(os.path.dirname(self.state_dict_path), network=self.runner.network)
            checkpoint_io.load(os.path.basename(self.state_dict_path))

        self.runner.py_logger.info("Start evaluation")

        if hasattr(self, 'data') and self.data is not None:
            dataset = DataLoader(self.data, batch_size=1,
                                 pin_memory=self.pin_memory, num_workers=0)
            self.runner.py_logger.info("Starting to evaluate for dataset.")

            loss_value = 0
            eval_start_time = time.time()
            stepsInEpoch = 0
            lossEpoch = 0.0
            valuesInEpoch = 0

            for i, model_input in enumerate(dataset):
                for k,v in model_input.items():
                    if isinstance(v,torch.Tensor):
                        model_input[k] = v.cuda()
                # detach the input
                model_input["detach"] = True
                if not self.runner.perf_test:
                    self.runner.evaluator.epoch_hook(i, model_input)

                if self.evaluate_loss and hasattr(self.runner, 'loss'):
                    model_output = self.runner.network(model_input)
                    loss = self.runner.loss(model_output,model_input)
                    self.runner.evaluator.train_hook(model_output, loss)
                    loss_value = sum([v.mean() for v in loss.values()])

                    #prevent memory bug
                    del loss
                    del model_output
                    del model_input

                    loss_value = loss_value.detach()
                    self.runner.logger.log_scalar("loss",loss_value)
                    lossEpoch += loss_value.mean().detach().cpu()

                self.runner.logger.increment()
                stepsInEpoch += 1

            if not self.runner.perf_test:
                if hasattr(self.runner, 'loss'):
                    self.runner.logger.log_scalar("loss_final", lossEpoch/stepsInEpoch)
                    self.runner.py_logger.info(f"loss_final: {lossEpoch/stepsInEpoch}")
                self.runner.py_logger.info(f"eval finished")
            else:
                print(f"Evaluation of {stepsInEpoch} steps took {time.time()-eval_start_time}s")
        else:
            # evaluate mesh and levelset
            self.runner.evaluator.epoch_hook(0, {})
