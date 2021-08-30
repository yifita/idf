import json
from ssl import RAND_pseudo_bytes
from typing import Dict, Tuple, List
from runner import Runner
from task.task import Task
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from checkpoint_io import CheckpointIO
import bisect
from helper import get_class_from_string, slice_dict
import time

class Train(Task):

    def __init__(self, config):
        self.last_time :float = 0
        self.epochs : int = 1000
        self.learning_rate :float= 1e-4
        self.linear_growth : bool = True
        self.multi = False
        self.update_checkpoint :int = 100
        self.pin_memory : bool = True
        self.clip :float = None
        self.resume_from : str = 'Train_latest.ckpt'
        self.resume_epoch : int = None
        self.optimizer : str = "Adam"
        self.phase : Dict[str, List[Tuple[float, float]]] = {}  # {network: [[phase, lr_factor]...]]
        self.batch_size : int = 1
        self.force_run : bool = False
        self.old_train : bool = False
        super().__init__(config)

    def _get_lr_lambdas(self):
        """
        If lr_factor == -1, nothing is done,
        Args:
            progress:
        """
        lr_lambdas = []
        for name in self._param_groups:
            def func(epoch, name=name):
                progress = epoch / float(self.epochs)
                _phase = bisect.bisect_left(self._phase_progress[name], progress)
                if _phase >= len(self._phase_progress[name]):
                    return  self._phase_lr_factor[name][-1]

                if self.linear_growth and _phase > 0:
                    # cosine anealing
                    v0 = self._phase_lr_factor[name][_phase-1]
                    p1p0 = (self._phase_progress[name][_phase] - self._phase_progress[name][_phase-1])
                    pp0 = progress-self._phase_progress[name][_phase-1]
                    v1v0 = (self._phase_lr_factor[name][_phase] - self._phase_lr_factor[name][_phase-1])
                    return v0 + (1-np.cos(np.pi*pp0/p1p0)) * v1v0 / 2
                    # return v0 + pp0/p1p0*v1v0

                return self._phase_lr_factor[name][_phase]
            lr_lambdas.append(func)
        return lr_lambdas

    def _train_phase(self, scheduler):
        """ Set requires_grad to True/False based on give phase """
        for name, lr in dict(zip(self._param_groups, scheduler.get_lr())).items():
            if lr == 0:
                getattr(self.runner.network, name).requires_grad_(False)
                self.runner.py_logger.info(f'Froze parameters of {name}.')
            elif lr > 0:
                getattr(self.runner.network, name).requires_grad_(True)
                self.runner.py_logger.info(f'Unfroze parameters of {name}.')

    def perf_step(self, name = None):
        return
        c_time = time.perf_counter()
        delta = c_time-self.last_time
        self.last_time = c_time
        if(name is not None):
            self.runner.py_logger.info(f'Phase {name} took {delta}s')


    def __call__(self):
        super().__call__()

        if not hasattr(self, 'data') and hasattr(self.runner, 'data'):
            self.data = self.runner.data

        if not hasattr(self, 'loss') and hasattr(self.runner, 'loss'):
            self.loss = self.runner.loss

        self.runner.py_logger.info(json.dumps(Runner.filterdict(self.runner.key_value.copy()), indent=4, skipkeys=True, sort_keys=True))
        self.runner.logger.log_config()

        self.runner.py_logger.info(f"Start training for {self.epochs} epochs")
        # copy entire code to logs
        self.runner.logger.log_code_base()

        self.runner.network.cuda()

        Optim = get_class_from_string("torch.optim."+self.optimizer)
        self._param_groups = list(self.phase.keys()) if len(self.phase) > 0 else [name for name, _ in self.runner.network.named_children()]
        if self.old_train:
            optimizer = torch.optim.Adam(lr=1e-4,params=self.runner.network.parameters())
        else:
            optimizer = Optim(lr=self.learning_rate, params=[{'params': [p for p in getattr(self.runner.network, name).parameters()]} for name in self._param_groups], betas=(0.1, 0.9))

        self.ckpt_io = CheckpointIO(self.folder, network=self.runner.network, optimizer=optimizer)

        # force init
        len(self.data)
        dataset = DataLoader(self.data, batch_size=self.batch_size,
                             pin_memory=self.pin_memory, num_workers=1 if self.multi else 0)

        total_it = self.epochs * len(dataset)
        # resume
        try:
            load_dict = self.ckpt_io.load(self.resume_from)
        except Exception as e:
            self.runner.py_logger.warn(repr(e))
            load_dict = dict()

        it =  load_dict.get('it', 0)
        ep = load_dict.get('epoch', 0)

        if self.resume_epoch is not None:
            it = self.resume_epoch * len(dataset)
            ep = self.resume_epoch

        ep -= 1

        self._last_phase = {key: -1 for key in self._param_groups}

        self._phase_progress = {key: [values[0] for values in self.phase.get(key, [[1.0, 1.0],])] for key in self._param_groups}
        self._phase_lr_factor = {key: [values[1] for values in self.phase.get(key, [[1.0, 1.0],])] for key in self._param_groups}
        if not self.old_train:
            if ep > -1:
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', self.learning_rate)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._get_lr_lambdas(), last_epoch=ep)

        self.runner.logger.set_step(it)
        avg_time = 0.0
        min_time = float('inf')
        max_time = 0.0
        train_start_time = time.time()
        while True:
            start_time = time.time()

            ep += 1
            if ep > self.epochs:
                break
            if not self.runner.perf_test:
                if(ep > 0 and self.update_checkpoint > 0 and ep % self.update_checkpoint == 0):
                    self.runner.py_logger.info(f"Creating checkpoint for {ep}/{self.epochs}")
                    self.ckpt_io.save(f'Train_{ep}.ckpt', epoch=ep, it=it)
                    latest_path = self.folder + "/Train_latest.ckpt"
                    if(os.path.isfile(latest_path) or os.path.islink(latest_path)):
                        self.runner.py_logger.info(f"Deleteing checkpoint {latest_path}")
                        os.remove(latest_path)
                    os.symlink(f'./Train_{ep}.ckpt',latest_path)
            if not self.old_train:
                self._train_phase(scheduler)
            lossEpoch = 0
            stepsInEpoch = 0
            loss_value = 0

            progress = ep / self.epochs
            if not self.runner.perf_test:
                dataset.dataset.epoch_hook(progress)

            for model_input in dataset :
                it += 1
                if not self.runner.perf_test:
                    self.runner.logger.increment()

                for k,v in model_input.items():
                    if isinstance(v,torch.Tensor):
                        model_input[k] = v.cuda()

                # detach the input
                model_input["detach"] = True
                model_input["istrain"] = True
                model_input["epoch"] = ep + float(stepsInEpoch)/len(dataset)
                model_input["iteration"] = it
                model_input["progress"] = progress
                model_input["force_run"] = self.force_run

                model_output = self.runner.network(model_input)
                loss = self.loss(model_output, model_input)

                if not self.runner.perf_test:
                    self.runner.evaluator.train_hook(model_output, loss)

                loss_value = sum([v.mean() for v in loss.values()])
                optimizer.zero_grad()
                loss_value.backward()

                if( not self.runner.perf_test and self.clip is not None and self.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.runner.network.parameters(), self.clip)

                optimizer.step()

                loss_value = loss_value.detach().cpu().item()
                if not self.runner.perf_test:
                    self.runner.logger.log_scalar("loss", loss_value)

                lossEpoch += loss_value
                stepsInEpoch += 1
                del model_output
                del loss
                del loss_value

                # I need to save more often, maybe you can use "benchmark" flag
                if not self.runner.perf_test and it % 1000 == 0:
                   self.ckpt_io.save(f'Train_latest.ckpt', epoch=ep, it=it)

            if not self.old_train:
                scheduler.step()
                if not self.runner.perf_test:
                    self.runner.logger.log_scalar("lr", dict(zip(self._param_groups, scheduler.get_last_lr())))


            cur_time = time.time()-start_time
            avg_time += cur_time/self.epochs
            min_time = min(cur_time,min_time)
            max_time = max(cur_time,max_time)

            if not self.runner.perf_test:
                self.runner.py_logger.info(f"Epoch {ep} average loss {lossEpoch/stepsInEpoch}")
                self.runner.evaluator.epoch_hook(ep, slice_dict(model_input, [0]))
                self.runner.py_logger.info(f"Finished training for epoch {ep}/{self.epochs} iter {it}/{total_it}")
                self.runner.py_logger.info(f"Running for {time.time()-train_start_time}s epoch times : avg:{(avg_time*self.epochs)/(ep+1)} min:{min_time} max:{max_time}")

        if(self.runner.perf_test):
                print(f"Training took {time.time()-train_start_time}s")
                print(f"Epoch times avg:{avg_time} min:{min_time} max:{max_time}")
        if not self.runner.perf_test:
            self.ckpt_io.save(f'Train_latest.ckpt', epoch=ep, it=it)