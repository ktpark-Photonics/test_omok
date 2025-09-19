"""Gomoku (Omok) board and game logic utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Player = int  # 1 for black, -1 for white
Move = Tuple[int, int]


@dataclass
class OmokState:
    """Represents the state of an Omok (Gomoku) board."""

    size: int = 15
    win_length: int = 5
    board: np.ndarray = field(default_factory=lambda: np.zeros((15, 15), dtype=np.int8))
    current_player: Player = 1
    last_move: Optional[Move] = None
    winner: Optional[Player] = None
    move_history: List[Move] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.board.shape != (self.size, self.size):
            self.board = np.zeros((self.size, self.size), dtype=np.int8)

    def clone(self) -> "OmokState":
        return OmokState(
            size=self.size,
            win_length=self.win_length,
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            move_history=list(self.move_history),
        )

    def reset(self) -> None:
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None
        self.winner = None
        self.move_history.clear()

    def available_moves(self) -> List[Move]:
        if self.winner is not None:
            return []
        empty_positions = np.argwhere(self.board == 0)
        return [tuple(pos) for pos in empty_positions]

    def apply_move(self, move: Move) -> None:
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: position already occupied")
        if self.winner is not None:
            raise ValueError("Cannot play move on finished game")
        self.board[row, col] = self.current_player
        self.last_move = move
        self.move_history.append(move)
        if self._check_winner(row, col):
            self.winner = self.current_player
        elif not self.available_moves():
            self.winner = 0  # draw
        self.current_player *= -1

    def _check_winner(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        if player == 0:
            return False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= self.win_length:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: Player) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def is_full(self) -> bool:
        return not np.any(self.board == 0)

    def encode(self, perspective: Player) -> np.ndarray:
        """Returns a representation of the board from the given perspective."""
        return (self.board * perspective).astype(np.float32)

    def render(self) -> str:
        symbols = {1: "●", -1: "○", 0: "."}
        lines = []
        for row in self.board:
            lines.append(" ".join(symbols[int(val)] for val in row))
        return "\n".join(lines)


def play_game(
    state: OmokState,
    agent_black,
    agent_white,
    max_moves: Optional[int] = None,
) -> Tuple[Optional[Player], List[Move]]:
    """Plays a single game between two agents and returns the winner."""

    state = state.clone()
    max_moves = max_moves or state.size * state.size
    for _ in range(max_moves):
        current_agent = agent_black if state.current_player == 1 else agent_white
        move = current_agent.select_move(state)
        state.apply_move(move)
        if state.winner is not None:
            return state.winner, list(state.move_history)
    return 0, list(state.move_history)
