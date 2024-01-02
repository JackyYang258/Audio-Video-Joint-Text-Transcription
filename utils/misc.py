import os
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)


def init_env(cfg):
    np.random.seed(cfg.common.seed)
    torch.manual_seed(cfg.common.seed)
    torch.cuda.manual_seed(cfg.common.seed)

    verify_checkpoint_directory(cfg.checkpoint.save_dir)


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        os.remove(temp_file_path)