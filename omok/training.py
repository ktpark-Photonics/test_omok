"""Training utilities for self-play reinforcement learning."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import torch

from .agent import PolicyAgent
from .game import OmokState, play_game


@dataclass
class TrainingConfig:
    board_size: int = 9
    win_length: int = 5
    episodes: int = 200
    learning_rate: float = 1e-3
    epsilon: float = 0.2
    report_interval: int = 10
    rolling_window: int = 50


@dataclass
class TrainingMetrics:
    episode: int
    black_wins: int
    white_wins: int
    draws: int
    black_loss: float
    white_loss: float
    black_win_rate: float
    white_win_rate: float
    best_black_win_rate: float
    best_white_win_rate: float
    timestamp: float


@dataclass
class TrainingResult:
    agent_black: PolicyAgent
    agent_white: PolicyAgent
    metrics: List[TrainingMetrics]


def train_self_play(
    config: TrainingConfig,
    agent_black: Optional[PolicyAgent] = None,
    agent_white: Optional[PolicyAgent] = None,
    progress_callback: Optional[callable] = None,
) -> TrainingResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_black = agent_black or PolicyAgent(board_size=config.board_size, learning_rate=config.learning_rate, epsilon=config.epsilon, device=device)
    agent_white = agent_white or PolicyAgent(board_size=config.board_size, learning_rate=config.learning_rate, epsilon=config.epsilon, device=device)

    state = OmokState(size=config.board_size, win_length=config.win_length)
    metrics: List[TrainingMetrics] = []
    rolling_black_wins: Deque[int] = deque(maxlen=config.rolling_window)
    rolling_white_wins: Deque[int] = deque(maxlen=config.rolling_window)

    best_black = 0.0
    best_white = 0.0

    for episode in range(1, config.episodes + 1):
        state.reset()
        agent_black.begin_episode()
        agent_white.begin_episode()

        winner, _ = play_game(state, agent_black, agent_white)

        if winner == 1:
            black_reward, white_reward = 1.0, -1.0
            rolling_black_wins.append(1)
            rolling_white_wins.append(0)
        elif winner == -1:
            black_reward, white_reward = -1.0, 1.0
            rolling_black_wins.append(0)
            rolling_white_wins.append(1)
        else:
            black_reward = white_reward = 0.0
            rolling_black_wins.append(0)
            rolling_white_wins.append(0)

        black_loss = agent_black.update(black_reward)
        white_loss = agent_white.update(white_reward)

        black_win_rate = (sum(rolling_black_wins) / len(rolling_black_wins)) if rolling_black_wins else 0.0
        white_win_rate = (sum(rolling_white_wins) / len(rolling_white_wins)) if rolling_white_wins else 0.0

        best_black = max(best_black, black_win_rate)
        best_white = max(best_white, white_win_rate)

        if episode % config.report_interval == 0 or episode == config.episodes:
            metrics_entry = TrainingMetrics(
                episode=episode,
                black_wins=sum(rolling_black_wins),
                white_wins=sum(rolling_white_wins),
                draws=len(rolling_black_wins) - sum(rolling_black_wins) - sum(rolling_white_wins),
                black_loss=black_loss,
                white_loss=white_loss,
                black_win_rate=black_win_rate,
                white_win_rate=white_win_rate,
                best_black_win_rate=best_black,
                best_white_win_rate=best_white,
                timestamp=time.time(),
            )
            metrics.append(metrics_entry)
            if progress_callback:
                progress_callback(metrics_entry)

    return TrainingResult(agent_black=agent_black, agent_white=agent_white, metrics=metrics)
