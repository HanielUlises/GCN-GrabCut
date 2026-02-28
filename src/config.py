import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict


# ---------------------------------------------------------
# Path handling
# ---------------------------------------------------------

# This resolves to: GCN-Grabcut/src/config/
CONFIG_DIR = (Path(__file__).resolve().parent / "config").resolve()

if not CONFIG_DIR.exists():
    raise RuntimeError(f"CONFIG directory not found at: {CONFIG_DIR}")


# ---------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------

@dataclass
class ModelConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int
    dropout: float


@dataclass
class DatasetConfig:
    root: str
    split: str
    img_size: int
    superpixels: int
    compactness: float


@dataclass
class GrabCutConfig:
    iterations: int
    lambda_val: float
    k: int


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    device: str

def load_raw_config(name: str) -> Dict[str, Any]:
    """
    Load a single raw YAML configuration by name (e.g., `model`, `dataset`).
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_config(name: str, cls):
    """
    Convert a YAML config into a typed dataclass.
    Example:
        model_cfg = create_config("model", ModelConfig)
    """
    raw = load_raw_config(name)
    return cls(**raw)


def load_config() -> Dict[str, Any]:
    """
    Load ALL configuration sections at once and return them in a dictionary.
    This is the main entry point for notebooks and scripts.
    """
    return {
        "model": create_config("model", ModelConfig),
        "dataset": create_config("dataset", DatasetConfig),
        "grabcut": create_config("grabcut", GrabCutConfig),
        "training": create_config("training", TrainingConfig),
    }
