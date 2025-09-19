"""Learning agent implementation for Omok using a policy gradient approach."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from .game import Move, OmokState, Player


class OmokPolicy(nn.Module):
    """Simple neural network that maps board states to move logits."""

    def __init__(self, board_size: int, hidden_sizes: Sequence[int] = (256, 128)) -> None:
        super().__init__()
        input_dim = board_size * board_size
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class PolicyAgent:
    """Policy gradient based Omok agent."""

    board_size: int
    learning_rate: float = 1e-3
    device: Optional[torch.device] = None
    hidden_sizes: Sequence[int] = (256, 128)
    epsilon: float = 0.1
    _policy: OmokPolicy = field(init=False, repr=False)
    _optimizer: optim.Optimizer = field(init=False, repr=False)
    _log_probs: List[torch.Tensor] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy = OmokPolicy(self.board_size, self.hidden_sizes).to(self.device)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self.learning_rate)

    def parameters(self):
        return self._policy.parameters()

    def eval(self) -> None:
        self._policy.eval()

    def train(self) -> None:
        self._policy.train()

    def select_move(self, state: OmokState, deterministic: bool = False, record: bool = True) -> Move:
        legal_moves = state.available_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        state_tensor = (
            torch.from_numpy(state.encode(state.current_player).reshape(-1))
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        logits = self._policy(state_tensor).squeeze(0)
        mask = torch.full((state.size * state.size,), fill_value=-1e9, device=self.device)
        for row, col in legal_moves:
            mask[row * state.size + col] = 0.0
        masked_logits = logits + mask
        if deterministic or np.random.rand() > self.epsilon:
            move_index = torch.argmax(masked_logits).item()
            log_prob = torch.log_softmax(masked_logits, dim=0)[move_index]
        else:
            distribution = Categorical(logits=masked_logits)
            move_index = int(distribution.sample().item())
            log_prob = distribution.log_prob(torch.tensor(move_index, device=self.device))
        row, col = divmod(move_index, state.size)
        if (row, col) not in legal_moves:
            # Fallback to a random legal move to avoid invalid plays
            row, col = legal_moves[np.random.randint(len(legal_moves))]
            move_index = row * state.size + col
            log_prob = torch.log_softmax(masked_logits, dim=0)[move_index]
        if record:
            self._log_probs.append(log_prob)
        return row, col

    def begin_episode(self) -> None:
        self._log_probs.clear()

    def update(self, reward: float) -> float:
        if not self._log_probs:
            return 0.0
        loss = -reward * torch.stack(self._log_probs).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._log_probs.clear()
        return float(loss.item())

    def state_dict(self):
        return {
            "policy": self._policy.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        self._policy.load_state_dict(state_dict["policy"])
        self._optimizer.load_state_dict(state_dict["optimizer"])

    def update_hyperparameters(
        self,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        if learning_rate is not None:
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = learning_rate
            self.learning_rate = learning_rate
        if epsilon is not None:
            self.epsilon = epsilon

    def clone(self) -> "PolicyAgent":
        clone_agent = PolicyAgent(
            board_size=self.board_size,
            learning_rate=self.learning_rate,
            hidden_sizes=self.hidden_sizes,
            epsilon=self.epsilon,
            device=self.device,
        )
        clone_agent.load_state_dict(self.state_dict())
        return clone_agent

    def to(self, device: torch.device) -> "PolicyAgent":
        if device == self.device:
            return self
        self._policy.to(device)
        for state in self._optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        self._log_probs = [log_prob.to(device) for log_prob in self._log_probs]
        self.device = device
        return self
