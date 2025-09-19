"""Training utilities for league-style self-play reinforcement learning."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Optional, Tuple

import torch

from .agent import PolicyAgent
from .game import Move, Player, OmokState, play_game


@dataclass
class TrainingConfig:
    board_size: int = 9
    win_length: int = 5
    episodes: int = 200
    learning_rate: float = 1e-3
    epsilon: float = 0.2
    report_interval: int = 10
    rolling_window: int = 50
    num_agents: int = 2
    device: str = "auto"


@dataclass
class AgentStanding:
    """Snapshot of a single agent's performance during training."""

    agent_index: int
    wins: int
    losses: int
    draws: int
    total_games: int
    win_rate: float
    recent_win_rate: float
    best_recent_win_rate: float
    average_loss: float

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_index + 1,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.total_games,
            "win_rate": self.win_rate,
            "recent_win_rate": self.recent_win_rate,
            "best_recent": self.best_recent_win_rate,
            "avg_loss": self.average_loss,
        }


@dataclass
class TrainingMetrics:
    round_index: int
    standings: List[AgentStanding]
    timestamp: float
    top_agent: int
    top_recent_win_rate: float


@dataclass
class TrainingResult:
    agents: List[PolicyAgent]
    metrics: List[TrainingMetrics]
    standings: List[AgentStanding]


MoveCallback = Callable[
    [int, int, int, Optional[Move], Player, OmokState, Tuple[int, int]],
    None,
]


def _round_robin_schedule(num_agents: int) -> List[List[Tuple[int, int]]]:
    """Generates a round-robin schedule for the given number of agents."""

    players: List[Optional[int]] = list(range(num_agents))
    if num_agents % 2 == 1:
        players.append(None)
    total_players = len(players)
    rounds = total_players - 1
    schedule: List[List[Tuple[int, int]]] = []

    for round_idx in range(rounds):
        pairings: List[Tuple[int, int]] = []
        for idx in range(total_players // 2):
            a = players[idx]
            b = players[total_players - 1 - idx]
            if a is None or b is None:
                continue
            # alternate colors across rounds so everyone gets chances to play first
            if round_idx % 2 == 0:
                pairings.append((a, b))
            else:
                pairings.append((b, a))
        schedule.append(pairings)
        # rotate all players but the first one (circle method)
        players = [players[0]] + [players[-1]] + players[1:-1]
    return schedule


def resolve_device(device_preference: Optional[str] = None) -> torch.device:
    """Return a torch.device based on a preference string."""

    if device_preference is None or device_preference.lower() in {"auto", "default"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized = device_preference.lower()
    if normalized in {"cpu", "cpu:0"}:
        return torch.device("cpu")

    if normalized.startswith("cuda") or normalized.startswith("gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA가 활성화된 PyTorch 빌드가 필요합니다.")
        # Support formats like "cuda", "cuda:0", or "gpu:1"
        if ":" in normalized:
            _, index_str = normalized.split(":", 1)
            if not index_str:
                index = 0
            else:
                index = int(index_str)
        else:
            index = 0
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"요청한 CUDA 디바이스(cuda:{index})를 찾을 수 없습니다. 연결된 GPU 수: {torch.cuda.device_count()}"
            )
        return torch.device(f"cuda:{index}")

    # Fall back to PyTorch device parsing which will raise on invalid input
    return torch.device(device_preference)


def _prepare_agents(
    num_agents: int,
    config: TrainingConfig,
    existing_agents: Optional[Iterable[PolicyAgent]],
    device: torch.device,
) -> List[PolicyAgent]:
    agents: List[PolicyAgent]
    if existing_agents is None:
        agents = [
            PolicyAgent(
                board_size=config.board_size,
                learning_rate=config.learning_rate,
                epsilon=config.epsilon,
                device=device,
            )
            for _ in range(num_agents)
        ]
    else:
        agents = list(existing_agents)
        if len(agents) < num_agents:
            agents.extend(
                PolicyAgent(
                    board_size=config.board_size,
                    learning_rate=config.learning_rate,
                    epsilon=config.epsilon,
                    device=device,
                )
                for _ in range(num_agents - len(agents))
            )
        elif len(agents) > num_agents:
            agents = agents[:num_agents]
        for agent in agents:
            agent.update_hyperparameters(
                learning_rate=config.learning_rate,
                epsilon=config.epsilon,
            )
            agent.to(device)
    return agents


def train_self_play(
    config: TrainingConfig,
    agents: Optional[Iterable[PolicyAgent]] = None,
    progress_callback: Optional[Callable[[TrainingMetrics], None]] = None,
    move_callback: Optional[MoveCallback] = None,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    """Trains a pool of agents in a rotating round-robin league."""

    resolved_device = device or resolve_device(config.device)
    num_agents = max(2, config.num_agents)

    league_agents = _prepare_agents(num_agents, config, agents, resolved_device)
    schedule = _round_robin_schedule(num_agents)

    wins = [0] * num_agents
    losses = [0] * num_agents
    draws = [0] * num_agents
    loss_sums = [0.0] * num_agents
    update_counts = [0] * num_agents
    recent_results: List[Deque[int]] = [deque(maxlen=config.rolling_window) for _ in range(num_agents)]
    best_recent_win_rate = [0.0] * num_agents

    metrics: List[TrainingMetrics] = []

    for round_number in range(1, config.episodes + 1):
        pairings = schedule[(round_number - 1) % len(schedule)]

        for match_index, (agent_a_idx, agent_b_idx) in enumerate(pairings):
            state = OmokState(size=config.board_size, win_length=config.win_length)
            agent_a = league_agents[agent_a_idx]
            agent_b = league_agents[agent_b_idx]
            agent_a.begin_episode()
            agent_b.begin_episode()

            if move_callback:
                move_callback(
                    round_number,
                    match_index,
                    -1,
                    None,
                    0,
                    state.clone(),
                    (agent_a_idx, agent_b_idx),
                )

            last_snapshot: Optional[OmokState] = None

            def wrapped_move_callback(snapshot: OmokState, move: Move, player: Player, move_index: int) -> None:
                nonlocal last_snapshot
                last_snapshot = snapshot
                if move_callback:
                    move_callback(
                        round_number,
                        match_index,
                        move_index,
                        move,
                        player,
                        snapshot,
                        (agent_a_idx, agent_b_idx),
                    )

            winner, _ = play_game(
                state,
                agent_a,
                agent_b,
                move_callback=wrapped_move_callback,
            )

            if winner == 1:
                reward_a, reward_b = 1.0, -1.0
                wins[agent_a_idx] += 1
                losses[agent_b_idx] += 1
                recent_results[agent_a_idx].append(1)
                recent_results[agent_b_idx].append(-1)
            elif winner == -1:
                reward_a, reward_b = -1.0, 1.0
                wins[agent_b_idx] += 1
                losses[agent_a_idx] += 1
                recent_results[agent_a_idx].append(-1)
                recent_results[agent_b_idx].append(1)
            else:
                reward_a = reward_b = 0.0
                draws[agent_a_idx] += 1
                draws[agent_b_idx] += 1
                recent_results[agent_a_idx].append(0)
                recent_results[agent_b_idx].append(0)

            loss_a = agent_a.update(reward_a)
            loss_b = agent_b.update(reward_b)
            loss_sums[agent_a_idx] += loss_a
            loss_sums[agent_b_idx] += loss_b
            update_counts[agent_a_idx] += 1
            update_counts[agent_b_idx] += 1

            if move_callback and last_snapshot is not None:
                move_callback(
                    round_number,
                    match_index,
                    -2,
                    None,
                    winner or 0,
                    last_snapshot,
                    (agent_a_idx, agent_b_idx),
                )

        if round_number % config.report_interval == 0 or round_number == config.episodes:
            standings: List[AgentStanding] = []
            for idx in range(num_agents):
                total_games = wins[idx] + losses[idx] + draws[idx]
                win_rate = wins[idx] / total_games if total_games > 0 else 0.0
                recent_deque = recent_results[idx]
                if recent_deque:
                    wins_recent = sum(1 for result in recent_deque if result == 1)
                    recent_win_rate = wins_recent / len(recent_deque)
                else:
                    recent_win_rate = 0.0
                best_recent_win_rate[idx] = max(best_recent_win_rate[idx], recent_win_rate)
                average_loss = (
                    loss_sums[idx] / update_counts[idx]
                    if update_counts[idx] > 0
                    else 0.0
                )
                standings.append(
                    AgentStanding(
                        agent_index=idx,
                        wins=wins[idx],
                        losses=losses[idx],
                        draws=draws[idx],
                        total_games=total_games,
                        win_rate=win_rate,
                        recent_win_rate=recent_win_rate,
                        best_recent_win_rate=best_recent_win_rate[idx],
                        average_loss=average_loss,
                    )
                )

            standings.sort(key=lambda s: (s.win_rate, s.recent_win_rate), reverse=True)
            top_agent_index = standings[0].agent_index if standings else 0
            top_recent = standings[0].recent_win_rate if standings else 0.0

            metrics_entry = TrainingMetrics(
                round_index=round_number,
                standings=standings,
                timestamp=time.time(),
                top_agent=top_agent_index,
                top_recent_win_rate=top_recent,
            )
            metrics.append(metrics_entry)
            if progress_callback:
                progress_callback(metrics_entry)

    final_standings = sorted(
        [
            AgentStanding(
                agent_index=idx,
                wins=wins[idx],
                losses=losses[idx],
                draws=draws[idx],
                total_games=wins[idx] + losses[idx] + draws[idx],
                win_rate=(wins[idx] / (wins[idx] + losses[idx] + draws[idx]))
                if (wins[idx] + losses[idx] + draws[idx])
                else 0.0,
                recent_win_rate=(
                    sum(1 for result in recent_results[idx] if result == 1) / len(recent_results[idx])
                    if recent_results[idx]
                    else 0.0
                ),
                best_recent_win_rate=best_recent_win_rate[idx],
                average_loss=(
                    loss_sums[idx] / update_counts[idx]
                    if update_counts[idx] > 0
                    else 0.0
                ),
            )
            for idx in range(num_agents)
        ],
        key=lambda s: (s.win_rate, s.recent_win_rate),
        reverse=True,
    )

    return TrainingResult(agents=league_agents, metrics=metrics, standings=final_standings)


# Backwards compatibility alias
train_league = train_self_play
