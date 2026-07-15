import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GeneModel


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def lambda_tag(lambda_value: float) -> str:
    ###### This just formats the string so it will save. 2.2-->2p2, -1=m1, etc ######
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")

def apply_theta_threshold(theta: torch.Tensor, threshold: float) -> torch.Tensor:
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    out = np.asarray(theta, dtype=np.float64).copy()
    if threshold > 0:
        out[np.abs(out) < threshold] = 0
    return out


def build_data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    preferred = data_dir / f"{tissue}_{gene_type.lower()}_data_binary_{age}m_droplet_union.npy"
    if preferred.exists():
        return preferred
    legacy = data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type.upper()}.npy"
    return legacy


def build_model_path(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float) -> Path:
    return models_dir / f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lam)}.pt"


def load_pt_param(tissue: str, models_dir: str | Path, age: int, lam: float, gene_type: str = "TG", threshold=None):
    '''
        NOTE THAT THETA HERE IS NOT THRESHOLDED YET UNLESS YOU INCLUDE IT AS AN ARGUMENT

    '''
    p = build_model_path(Path(models_dir), tissue, age, gene_type, lam)

    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")
    payload = torch.load(p, map_location="cpu")
    theta = payload["theta"]
    m = payload["m"]

    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()

    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()

    if threshold:
        return apply_theta_threshold(theta, threshold=threshold), m

    return theta, m

def load_training_arrays(data_dir: Path, tissue: str, age: int, gene_type: str):
    ### This loads data in the from f"data_bin_filt_{tissue}_{age}m_{gene_type}.npy" with gene type lowercase ######
    gt = gene_type.lower()
    data_path = build_data_path(data_dir, tissue, age, gt)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    data = np.load(data_path)
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {data_path}, got shape {data.shape}")

    if gt == "tf":
        tf_data = data
    else:
        tf_path = build_data_path(data_dir, tissue, age, "tf")
        if not tf_path.exists():
            raise FileNotFoundError(f"Missing TF reference file for TG training: {tf_path}")
        tf_data = np.load(tf_path)
        if tf_data.ndim != 2:
            raise ValueError(f"Expected 2D matrix in {tf_path}, got shape {tf_data.shape}")

    return tf_data, data, data_path

def load_data(data_dir, tissue, age, gene_type):
    data_path = build_data_path(data_dir, tissue, age, gene_type)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    tf_data = np.load(data_path)
    if tf_data.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {data_path}, got shape {tf_data.shape}")

    return tf_data, data_path

def load_payload(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return torch.load(path, map_location="cpu")

def load_theta(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float, threshold: float) -> np.ndarray:
    path = build_model_path(models_dir, tissue, age, gene_type, lam)
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    payload = load_payload(path)
    theta_all = payload["theta"]
    theta = apply_theta_threshold(theta_all, threshold)
    return theta


def load_m(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float, threshold: float) -> np.ndarray:
    path = build_model_path(models_dir, tissue, age, gene_type, lam)

    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")

    payload = load_payload(path)
    m = payload["m"]
    return m
