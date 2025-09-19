"""Omok training package."""

from .agent import PolicyAgent
from .game import OmokState, play_game
from .training import TrainingConfig, TrainingMetrics, TrainingResult, train_self_play

__all__ = [
    "PolicyAgent",
    "OmokState",
    "play_game",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResult",
    "train_self_play",
]
