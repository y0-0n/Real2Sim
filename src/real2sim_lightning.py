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
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)

def train(cfg: DictConfig) -> None:#Tuple[dict, dict]:
    ######################################
    # state estimation model (simulator parameter update)
    ######################################
    model = hydra.utils.instantiate(cfg.model)

    ######################################
    # real-world trajectory (online / offline)
    ######################################
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    ######################################
    # train
    ######################################
    
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))

    logging.info('Finish')

@hydra.main(version_base=None, config_path=root / "configs", config_name="real2sim")
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()
