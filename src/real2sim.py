import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    # indicator=[".git", ".gitignore", "README.md"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)

def train() -> None:#Tuple[dict, dict]:
    ######################################
    # state estimation model (simulator parameter update)
    ######################################
    model = hydra.utils.instantiate(cfg.model)

    ######################################
    # real-world trajectory (online / offline)
    ######################################

    ######################################
    # train
    ######################################
    
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))

    logging.info('Finish')

def main() -> None:
    # print(OmegaConf.to_yaml(cfg))
    train()

if __name__ == "__main__":
    main()
