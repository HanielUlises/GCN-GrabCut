import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_raw_config(name: str) -> Dict[str, Any]:
    """
    Load raw yaml configuration without parsing into dataclasses.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

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

def create_config(name: str, cls):
    """
    Convert a YAML config into a typed dataclass.
    Example:
        model_cfg = create_config("model", ModelConfig)
    """
    raw = load_raw_config(name)
    return cls(**raw)