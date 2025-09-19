"""Utilities for saving and loading Omok agent checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from .agent import PolicyAgent
from .training import TrainingConfig


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


@dataclass
class CheckpointMetadata:
    name: str
    created_at: str
    board_size: int
    win_length: int
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def save_checkpoint(
    path: Path,
    agent_black: PolicyAgent,
    agent_white: PolicyAgent,
    config: TrainingConfig,
    notes: Optional[str] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "black": agent_black.state_dict(),
        "white": agent_white.state_dict(),
        "config": {
            "board_size": config.board_size,
            "win_length": config.win_length,
            "learning_rate": config.learning_rate,
            "epsilon": config.epsilon,
        },
        "metadata": {
            "name": path.stem,
            "created_at": datetime.utcnow().isoformat(),
            "notes": notes or "",
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> Dict[str, object]:
    payload = torch.load(path, map_location=device or torch.device("cpu"))
    return payload


def list_checkpoints(directory: Path = CHECKPOINT_DIR):
    return sorted(directory.glob("*.pt"))
