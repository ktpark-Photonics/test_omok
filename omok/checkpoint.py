"""Utilities for saving and loading Omok agent checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

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
    agents: Sequence[PolicyAgent],
    config: TrainingConfig,
    standings: Optional[Iterable] = None,
    notes: Optional[str] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent_states = [agent.state_dict() for agent in agents]
    payload = {
        "config": {
            "board_size": config.board_size,
            "win_length": config.win_length,
            "learning_rate": config.learning_rate,
            "epsilon": config.epsilon,
            "num_agents": len(agents),
            "device": config.device,
        },
        "metadata": {
            "name": path.stem,
            "created_at": datetime.utcnow().isoformat(),
            "notes": notes or "",
        },
        "agents": agent_states,
    }
    if len(agent_states) >= 2:
        payload["black"] = agent_states[0]
        payload["white"] = agent_states[1]
    if standings:
        payload["standings"] = [
            entry.__dict__ if hasattr(entry, "__dict__") else dict(entry)
            for entry in standings
        ]
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> Dict[str, object]:
    payload = torch.load(path, map_location=device or torch.device("cpu"))
    return payload


def list_checkpoints(directory: Path = CHECKPOINT_DIR):
    return sorted(directory.glob("*.pt"))
