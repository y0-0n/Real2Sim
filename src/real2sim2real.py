import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    # indicator=[".git", ".gitignore", "README.md"],
    pythonpath=True,
    dotenv=True,
)

from utils.dataloader import load
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)

def train(cfg: DictConfig) -> None:#Tuple[dict, dict]:
    ######################################
    # environment
    ######################################
    env = hydra.utils.instantiate(cfg.environment)

    ######################################
    # model
    ######################################
    model = hydra.utils.instantiate(cfg.model)

    ######################################
    # real-world trajectory (online / offline)
    ######################################

    ######################################
    # state estimation (simulator parameter update)
    ######################################

    ######################################
    # 
    ######################################
    logging.info('Test')

@hydra.main(version_base=None, config_path=root / "configs", config_name="real2sim2real")
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()
