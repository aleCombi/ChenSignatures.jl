# common.py
"""Shared utilities for benchmark suite"""

import ast
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# -------- Constants --------

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.yaml"

# -------- Config Loading --------

def load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Simple YAML parser for benchmark config files"""
    cfg = {}
    if not path.is_file():
        return cfg
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue

            if value.startswith('"') and value.endswith('"'):
                cfg[key] = value[1:-1]
            elif value.startswith("["):
                try:
                    cfg[key] = ast.literal_eval(value)
                except:
                    cfg[key] = value
            else:
                try:
                    cfg[key] = int(value)
                except ValueError:
                    cfg[key] = value
    return cfg

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and parse benchmark configuration"""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = load_simple_yaml(config_path)

    Ns = raw.get("Ns", [150, 1000, 2000])
    Ds = raw.get("Ds", [2, 6, 7, 8])
    Ms = raw.get("Ms", [4, 6])
    path_kind = raw.get("path_kind", "linear")
    runs_dir = raw.get("runs_dir", "runs")
    repeats = int(raw.get("repeats", 5))
    logsig_method = raw.get("logsig_method", "O")
    operations = raw.get("operations", ["signature", "logsignature"])

    path_kind = path_kind.lower()
    if path_kind not in ("linear", "sin"):
        raise ValueError(f"Unknown path_kind '{path_kind}', expected 'linear' or 'sin'.")

    return {
        "Ns": Ns,
        "Ds": Ds,
        "Ms": Ms,
        "path_kind": path_kind,
        "runs_dir": runs_dir,
        "repeats": repeats,
        "logsig_method": logsig_method,
        "operations": operations,
    }

# -------- Path Generators --------

def make_path_linear(d: int, N: int) -> np.ndarray:
    """Generate linear path: [t, 2t, 2t, ...]"""
    ts = np.linspace(0.0, 1.0, N)
    path = np.empty((N, d), dtype=float)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path

def make_path_sin(d: int, N: int) -> np.ndarray:
    """Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]"""
    ts = np.linspace(0.0, 1.0, N)
    omega = 2.0 * math.pi
    # Matches Julia: path[i, k] = sin(2pi * t * k)
    # Python array is 0-indexed, so k=1..d maps to cols 0..d-1
    ks = np.arange(1, d + 1, dtype=float)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path

def make_path(d: int, N: int, kind: str) -> np.ndarray:
    """Generate path of specified kind"""
    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")