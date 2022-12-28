import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    # indicator=[".git", ".gitignore", "README.md"],
    pythonpath=True,
    dotenv=True,
)

from typing import Any, List
# import sys
# print(sys.path)

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import ray
import time
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import deque
from torch.utils.data import DataLoader
from src.utils.ReplayBuffer import ReplayBuffer
from src.envs.mujoco.Ant import AntRandomEnvClass

@ray.remote
class Worker(object):
    def __init__(self,
                env,
                x_minmax=np.array([[0, 1]]),
                worker_id=0) -> None:
        self.x_minmax = x_minmax # shape (num_parameter, 2)
        self.worker_id = worker_id

        self.env = env(VERBOSE=False, rand_mass=[1,2],rand_fric=[0.3, 0.8],render_mode=None)       

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def simulate(self, x):
        # self.logger.warning(self.worker_id)
        env = self.env
        for i in range(300):
            env.render()
            if i % 100 == 0:
                env.reset()
                env.reset_random()
                # env.set_box_weight(1)
                # env.set_leg_weight(1)
                # env.set_fric(0.2)
                # env.set_box_weight(1)
                # print(env.get_fric(), env.get_box_weight(), env.get_leg_weight())
            action = np.random.standard_normal(8) * 0.7
            env.step(action)
class BOModule(object):
    def __init__(
        self,
        env,
        n_workers:int=2,
        batch_size:int=5,
        x_minmax:np.array=np.array([[1e3,1e5]]),
        n_sample_max:int=1000
    ):
        super().__init__()

        ######################################
        # environment
        ######################################
        # env = hydra.utils.instantiate(environment)
        self.x_data = deque()
        self.y_data = deque()
        self.batch_size = batch_size
        self.x_minmax = x_minmax
        self.n_workers = n_workers
        # print(environment)
        # self.envs = [hydra.utils.instantiate(cfg.environment) for i in range(self.n_workers)]
        self.workers = [Worker.remote(env=env, worker_id=i) for i in range(n_workers)]

    def sample(self,n_sample):
        """
        Sample x as a list from the input domain 
        """
        x_samples = []
        for _ in range(n_sample):
            x_samples.append(
                self.x_minmax[:,0]+(self.x_minmax[:,1]-self.x_minmax[:,0])*np.random.rand(1,self.x_minmax.shape[0]))
        return x_samples

    def evaluation(self):
        """
        """
        # print(self.sample(5))
        x_evals = self.sample(self.n_workers)
        evals = [self.workers[i].simulate.remote(x=x_eval) for i, x_eval in enumerate(x_evals)]
        trajs = ray.get(evals)
        print(trajs)



if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    # _ = hydra.utils.instantiate(cfg)
    module = BOModule(env=AntRandomEnvClass)
    module.evaluation()



