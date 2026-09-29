
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def collate_pairs(batch):
    drug_embedding, protein_embedding, drug_mask, protein_mask, labels = zip(*batch)
    return (
        (torch.cat(drug_embedding, dim=0), torch.cat(drug_mask, dim=0)),
        (torch.cat(protein_embedding, dim=0), torch.cat(protein_mask, dim=0)),
        torch.tensor(labels, dtype=torch.long),
    )


def save_best_checkpoint(state: Dict, checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "model_best.pth"
    torch.save(state, path)
    return path


def create_logger(directory: Path) -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("CFGBDTI.{}.{}".format(directory, time.time()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
