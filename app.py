"""Streamlit app for training and playing an Omok reinforcement learning agent."""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

from omok.agent import PolicyAgent
from omok.checkpoint import CHECKPOINT_DIR, list_checkpoints, load_checkpoint, save_checkpoint
from omok.game import Move, OmokState
from omok.training import TrainingConfig, TrainingMetrics, train_self_play

st.set_page_config(page_title="Omok RL Trainer", layout="wide")

CUSTOM_CSS = """
<style>
:root {
    --omok-wood-light: #f6dfb4;
    --omok-wood-dark: #ebc988;
    --omok-line: rgba(120, 72, 20, 0.45);
    --omok-highlight: rgba(255, 176, 58, 0.55);
}
.stApp {
    background: radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.4)), #fbf7ef;
}
.omok-preview {
    display: inline-block;
    padding: 1rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.2)), var(--omok-wood-light);
    border-radius: 1rem;
    box-shadow: 0 12px 28px rgba(74, 42, 5, 0.18);
    border: 1px solid rgba(138, 94, 45, 0.35);
}
.omok-preview table {
    border-collapse: collapse;
    margin: 0 auto;
}
.omok-preview th {
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.25rem 0.35rem;
    color: #6a3f0b;
    text-align: center;
}
.omok-preview td {
    width: 2.6rem;
    height: 2.6rem;
    border: 1px solid var(--omok-line);
    text-align: center;
    vertical-align: middle;
    font-size: 1.6rem;
    background: radial-gradient(circle at center, rgba(0, 0, 0, 0.1) 8%, transparent 9%), var(--omok-wood-dark);
    color: #111;
}
.omok-preview td.white {
    color: #f8f8f8;
    text-shadow: 0 0 6px rgba(0, 0, 0, 0.55);
}
.omok-preview td.black {
    color: #111;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.35);
}
.omok-preview td.empty {
    color: rgba(0, 0, 0, 0);
}
.omok-preview td.highlight {
    box-shadow: inset 0 0 0 3px var(--omok-highlight);
    background: radial-gradient(circle at center, rgba(255, 215, 0, 0.35) 20%, transparent 50%), var(--omok-wood-dark);
}
.omok-preview .board-caption {
    margin-top: 0.75rem;
    text-align: center;
    font-size: 0.85rem;
    color: #6a3f0b;
    font-weight: 600;
}
.omok-interactive div[data-testid="column"] {
    padding: 0.05rem 0.1rem !important;
}
.omok-interactive div[data-testid="stButton"] > button {
    width: 100%;
    aspect-ratio: 1;
    border-radius: 0.35rem;
    border: 1px solid var(--omok-line);
    background: radial-gradient(circle at center, rgba(0, 0, 0, 0.1) 8%, transparent 9%), var(--omok-wood-dark);
    font-size: 1.55rem;
    color: #111;
    transition: transform 0.05s ease-in-out, box-shadow 0.1s ease-in-out;
}
.omok-interactive div[data-testid="stButton"] > button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 10px 18px rgba(0, 0, 0, 0.12);
}
.omok-interactive div[data-testid="stButton"] > button:disabled {
    opacity: 0.85;
    color: #111;
}
.omok-interactive .coord-label {
    text-align: center;
    font-weight: 600;
    color: #6a3f0b;
    padding: 0.2rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    if "training_history" not in st.session_state:
        st.session_state["training_history"] = []
    if "agents" not in st.session_state:
        st.session_state["agents"] = None
    if "play_state" not in st.session_state:
        st.session_state["play_state"] = OmokState(size=9, win_length=5)
    if "play_message" not in st.session_state:
        st.session_state["play_message"] = "에이전트와 대국을 시작해보세요!"
    if "play_user_color" not in st.session_state:
        st.session_state["play_user_color"] = "black"
    if "active_checkpoint" not in st.session_state:
        st.session_state["active_checkpoint"] = None


def board_preview_html(state: OmokState, highlight: Optional[Move] = None, caption: Optional[str] = None) -> str:
    size = state.size
    highlight = tuple(highlight) if highlight is not None else None
    header_cells = "".join(f"<th>{idx}</th>" for idx in range(1, size + 1))
    header = f"<tr><th></th>{header_cells}</tr>"
    rows = []
    for row in range(size):
        cells = [f"<th>{row + 1}</th>"]
        for col in range(size):
            value = state.board[row, col]
            classes = ["cell"]
            if value == 1:
                classes.append("black")
                symbol = "●"
            elif value == -1:
                classes.append("white")
                symbol = "○"
            else:
                classes.append("empty")
                symbol = ""
            if highlight == (row, col):
                classes.append("highlight")
            class_name = " ".join(classes)
            cells.append(f"<td class=\"{class_name}\">{symbol}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    caption_html = f"<div class='board-caption'>{caption}</div>" if caption else ""
    return "".join(
        [
            "<div class='omok-preview'>",
            "<table>",
            header,
            "".join(rows),
            "</table>",
            caption_html,
            "</div>",
        ]
    )


def ensure_agents(config: TrainingConfig) -> Dict[str, object]:
    agents_bundle = st.session_state.get("agents")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not agents_bundle or agents_bundle["config"].board_size != config.board_size:
        agent_black = PolicyAgent(board_size=config.board_size, learning_rate=config.learning_rate, epsilon=config.epsilon, device=device)
        agent_white = PolicyAgent(board_size=config.board_size, learning_rate=config.learning_rate, epsilon=config.epsilon, device=device)
        st.session_state["agents"] = {
            "black": agent_black,
            "white": agent_white,
            "config": config,
        }
    else:
        agents_bundle["config"] = config
        agents_bundle["black"].update_hyperparameters(learning_rate=config.learning_rate, epsilon=config.epsilon)
        agents_bundle["white"].update_hyperparameters(learning_rate=config.learning_rate, epsilon=config.epsilon)
    return st.session_state["agents"]


def update_training_history(metric: TrainingMetrics, placeholder_chart, placeholder_metrics):
    st.session_state["training_history"].append(metric)
    history_df = pd.DataFrame([
        {
            "episode": m.episode,
            "black_win_rate": m.black_win_rate,
            "white_win_rate": m.white_win_rate,
            "black_loss": m.black_loss,
            "white_loss": m.white_loss,
            "best_black": m.best_black_win_rate,
            "best_white": m.best_white_win_rate,
        }
        for m in st.session_state["training_history"]
    ])
    history_df.set_index("episode", inplace=True)
    placeholder_chart.line_chart(history_df[["black_win_rate", "white_win_rate"]])
    with placeholder_metrics.container():
        col_a, col_b = st.columns(2)
        col_a.metric("최근 블랙 평균 승률", f"{metric.black_win_rate:.2f}", delta=f"최고 {metric.best_black_win_rate:.2f}")
        col_b.metric("최근 화이트 평균 승률", f"{metric.white_win_rate:.2f}", delta=f"최고 {metric.best_white_win_rate:.2f}")


def run_training_ui():
    st.header("🔁 자기대국 학습")
    st.write("두 에이전트가 서로 대국하며 정책을 학습합니다. 학습이 진행되는 동안 실시간으로 승률과 손실을 확인할 수 있습니다.")

    with st.form("training_form"):
        board_size = st.slider("바둑판 크기", min_value=5, max_value=15, value=9)
        episodes = st.number_input("에피소드 수", min_value=10, max_value=100000, value=200, step=10)
        learning_rate = st.number_input("학습률", min_value=1e-4, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")
        epsilon = st.slider("탐험 비율 (epsilon)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        report_interval = st.number_input("리포트 주기", min_value=1, max_value=200, value=20)
        rolling_window = st.number_input("승률 이동 평균 길이", min_value=5, max_value=200, value=50)
        show_training_visual = st.checkbox("학습 대국 실시간 보기", value=True)
        visual_delay = st.slider("수 시각화 지연 (초)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        submitted = st.form_submit_button("학습 시작")

    if submitted:
        config = TrainingConfig(
            board_size=int(board_size),
            win_length=5,
            episodes=int(episodes),
            learning_rate=float(learning_rate),
            epsilon=float(epsilon),
            report_interval=int(report_interval),
            rolling_window=int(rolling_window),
        )
        agents_bundle = ensure_agents(config)
        if not show_training_visual:
            visual_delay = 0.0

        if show_training_visual:
            col_visual, col_metrics = st.columns([1, 2])
            with col_visual:
                st.caption("실시간 자기대국")
                board_placeholder = st.empty()
                move_placeholder = st.empty()
                board_placeholder.markdown(
                    board_preview_html(
                        OmokState(size=config.board_size, win_length=config.win_length),
                        caption="대국 준비중",
                    ),
                    unsafe_allow_html=True,
                )
                move_placeholder.caption("학습 수를 기다리는 중...")
        else:
            col_metrics = st.container()
            board_placeholder = None
            move_placeholder = None

        with col_metrics:
            placeholder_chart = st.empty()
            placeholder_metrics = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_reports = max(1, math.ceil(config.episodes / config.report_interval))
        reports_seen = 0

        def visualize_move(episode_idx: int, move_idx: int, move: Move, player: int, snapshot: OmokState) -> None:
            if board_placeholder is None:
                return
            color = "흑" if player == 1 else "백"
            caption = f"에피소드 {episode_idx:,} / {config.episodes:,} · {color}"
            board_placeholder.markdown(
                board_preview_html(snapshot, highlight=move, caption=caption),
                unsafe_allow_html=True,
            )
            move_placeholder.markdown(
                f"**최근 수:** {color} ({move[0] + 1}, {move[1] + 1})",
                unsafe_allow_html=True,
            )
            if visual_delay > 0:
                time.sleep(visual_delay)

        def progress_callback(metric: TrainingMetrics):
            nonlocal reports_seen
            reports_seen += 1
            update_training_history(metric, placeholder_chart, placeholder_metrics)
            progress_bar.progress(min(1.0, reports_seen / total_reports))
            status_text.info(
                f"에피소드 {metric.episode} / {config.episodes} - 블랙 승률 {metric.black_win_rate:.2f}, 화이트 승률 {metric.white_win_rate:.2f}"
            )

        start_time = time.time()
        result = train_self_play(
            config,
            agent_black=agents_bundle["black"],
            agent_white=agents_bundle["white"],
            progress_callback=progress_callback,
            move_callback=visualize_move if show_training_visual else None,
        )
        progress_bar.progress(1.0)
        elapsed = time.time() - start_time
        st.success(f"학습이 완료되었습니다! (소요 시간: {elapsed:.1f}초)")
        st.session_state["agents"] = {
            "black": result.agent_black,
            "white": result.agent_white,
            "config": config,
        }
        st.session_state["active_checkpoint"] = None

    if st.session_state["training_history"]:
        st.subheader("📈 최근 학습 지표")
        history_df = pd.DataFrame([
            {
                "episode": m.episode,
                "black_win_rate": m.black_win_rate,
                "white_win_rate": m.white_win_rate,
                "black_loss": m.black_loss,
                "white_loss": m.white_loss,
                "best_black": m.best_black_win_rate,
                "best_white": m.best_white_win_rate,
            }
            for m in st.session_state["training_history"]
        ])
        st.dataframe(history_df.set_index("episode"))


def get_active_agent(color: str) -> Optional[PolicyAgent]:
    agents_bundle = st.session_state.get("agents")
    if not agents_bundle:
        return None
    return agents_bundle[color]


def agent_move(state: OmokState, color: str) -> None:
    agent = get_active_agent(color)
    if agent is None:
        st.warning("학습된 에이전트가 없습니다. 먼저 학습을 진행하거나 체크포인트를 불러오세요.")
        return
    expected_player = 1 if color == "black" else -1
    if state.current_player != expected_player or state.winner is not None:
        return
    agent.begin_episode()
    move = agent.select_move(state, deterministic=True, record=False)
    state.apply_move(move)


def render_board(state: OmokState):
    size = state.size
    user_color = st.session_state["play_user_color"]
    user_turn = 1 if user_color == "black" else -1
    clicked_move: Optional[Tuple[int, int]] = None

    board_container = st.container()
    with board_container:
        st.markdown("<div class='omok-preview omok-interactive'>", unsafe_allow_html=True)
        header_cols = st.columns([0.8] + [1] * size)
        header_cols[0].markdown("&nbsp;", unsafe_allow_html=True)
        for idx, col in enumerate(header_cols[1:], start=1):
            col.markdown(f"<div class='coord-label'>{idx}</div>", unsafe_allow_html=True)

        for row in range(size):
            row_cols = st.columns([0.8] + [1] * size)
            row_cols[0].markdown(f"<div class='coord-label'>{row + 1}</div>", unsafe_allow_html=True)
            for col in range(size):
                cell_value = state.board[row, col]
                label = "●" if cell_value == 1 else "○" if cell_value == -1 else ""
                disabled = (
                    cell_value != 0
                    or state.winner is not None
                    or state.current_player != user_turn
                )
                if (
                    row_cols[col + 1].button(
                        label or " ",
                        key=f"cell_{row}_{col}",
                        use_container_width=True,
                        disabled=disabled,
                    )
                    and clicked_move is None
                ):
                    clicked_move = (row, col)
        st.markdown("</div>", unsafe_allow_html=True)

    if clicked_move is not None:
        state.apply_move(clicked_move)
        if state.winner is not None:
            if state.winner == 0:
                st.session_state["play_message"] = "무승부입니다."
            elif (state.winner == 1 and user_color == "black") or (state.winner == -1 and user_color == "white"):
                st.session_state["play_message"] = "축하합니다! 당신이 승리했습니다."
            else:
                st.session_state["play_message"] = "아쉽네요! 에이전트가 승리했습니다."
            return

        agent_color = "white" if user_color == "black" else "black"
        agent_move(state, agent_color)
        if state.winner is None:
            st.session_state["play_message"] = "당신의 차례입니다."
        else:
            if state.winner == 0:
                st.session_state["play_message"] = "무승부입니다."
            elif (state.winner == 1 and user_color == "black") or (state.winner == -1 and user_color == "white"):
                st.session_state["play_message"] = "축하합니다! 당신이 승리했습니다."
            else:
                st.session_state["play_message"] = "아쉽네요! 에이전트가 승리했습니다."


def run_play_ui():
    st.header("🎮 에이전트와 대국하기")
    st.write("학습된 에이전트와 직접 오목을 두어보세요. 원하는 체크포인트를 선택해서 플레이할 수 있습니다.")

    state = st.session_state["play_state"]
    agents_bundle = st.session_state.get("agents")
    if agents_bundle and agents_bundle["config"].board_size != state.size:
        state = OmokState(size=agents_bundle["config"].board_size, win_length=5)
        st.session_state["play_state"] = state

    col_settings, col_board = st.columns([1, 2])

    with col_settings:
        st.write("**플레이 설정**")
        color = st.radio(
            "내 돌 색",
            options=["black", "white"],
            format_func=lambda x: "흑 (●)" if x == "black" else "백 (○)",
            index=0 if st.session_state["play_user_color"] == "black" else 1,
        )
        if color != st.session_state["play_user_color"]:
            st.session_state["play_user_color"] = color
            st.session_state["play_state"] = OmokState(size=state.size, win_length=5)
            state = st.session_state["play_state"]
            if color == "white":
                agent_move(state, "black")
                if state.winner is None:
                    st.session_state["play_message"] = "당신의 차례입니다."
            else:
                st.session_state["play_message"] = "당신이 선입니다."

        if st.button("새 게임 시작"):
            st.session_state["play_state"] = OmokState(size=state.size, win_length=5)
            st.session_state["play_message"] = "새 게임이 시작되었습니다!"
            state = st.session_state["play_state"]
            if st.session_state["play_user_color"] == "white":
                agent_move(state, "black")
                if state.winner is None:
                    st.session_state["play_message"] = "당신의 차례입니다."
            else:
                st.session_state["play_message"] = "당신이 선입니다."

        st.write("현재 선택된 체크포인트:")
        if st.session_state["active_checkpoint"]:
            st.caption(st.session_state["active_checkpoint"])
        else:
            st.caption("(현재 메모리에 있는 에이전트 사용)")

    with col_board:
        st.write("**게임 보드**")
        render_board(state)
        st.markdown(
            board_preview_html(state, highlight=state.last_move, caption="현재 판"),
            unsafe_allow_html=True,
        )
        if state.last_move is not None:
            if state.winner in (1, -1):
                last_player = state.winner
            else:
                last_player = -state.current_player
            color_name = "흑" if last_player == 1 else "백"
            move_row, move_col = state.last_move
            st.caption(f"마지막 수: {color_name} ({move_row + 1}, {move_col + 1})")
        if state.winner is not None:
            if state.winner == 0:
                st.success("무승부입니다!")
            elif state.winner == 1:
                st.success("흑이 승리했습니다!")
            else:
                st.success("백이 승리했습니다!")

    st.info(st.session_state["play_message"])


def run_checkpoint_ui():
    st.header("💾 체크포인트 관리")
    st.write("학습한 에이전트의 파라미터를 저장하거나 불러옵니다. 저장된 체크포인트는 왼쪽 패널에서 간편하게 선택할 수 있습니다.")

    agents_bundle = st.session_state.get("agents")

    with st.form("save_checkpoint"):
        checkpoint_name = st.text_input("체크포인트 이름", value=f"omok_{int(time.time())}")
        notes = st.text_area("비고", placeholder="체크포인트에 대한 메모를 남길 수 있습니다.")
        submitted = st.form_submit_button("체크포인트 저장")
        if submitted:
            if not agents_bundle:
                st.error("저장할 에이전트가 없습니다. 먼저 학습을 진행하세요.")
            else:
                config: TrainingConfig = agents_bundle["config"]
                path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
                save_checkpoint(path, agents_bundle["black"], agents_bundle["white"], config, notes=notes)
                st.success(f"체크포인트가 저장되었습니다: {path}")
                st.session_state["active_checkpoint"] = str(path)

    st.subheader("📂 저장된 체크포인트")
    checkpoint_files = list_checkpoints()
    if not checkpoint_files:
        st.info("저장된 체크포인트가 없습니다. 먼저 학습을 진행하거나 새로운 체크포인트를 생성해보세요.")
    for ckpt_path in checkpoint_files:
        payload = load_checkpoint(ckpt_path)
        config = payload.get("config", {})
        metadata = payload.get("metadata", {})
        with st.expander(f"{metadata.get('name', ckpt_path.stem)} ({config.get('board_size', '?')}x{config.get('board_size', '?')})"):
            st.caption(f"생성 시각: {metadata.get('created_at', '알 수 없음')}")
            if metadata.get("notes"):
                st.write(metadata["notes"])
            if st.button("이 체크포인트 불러오기", key=f"load_{ckpt_path.stem}"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                config_data = payload.get("config", {})
                config_obj = TrainingConfig(
                    board_size=config_data.get("board_size", 9),
                    win_length=config_data.get("win_length", 5),
                    learning_rate=config_data.get("learning_rate", 1e-3),
                    epsilon=config_data.get("epsilon", 0.2),
                )
                agent_black = PolicyAgent(board_size=config_obj.board_size, learning_rate=config_obj.learning_rate, epsilon=config_obj.epsilon, device=device)
                agent_white = PolicyAgent(board_size=config_obj.board_size, learning_rate=config_obj.learning_rate, epsilon=config_obj.epsilon, device=device)
                agent_black.load_state_dict(payload["black"])
                agent_white.load_state_dict(payload["white"])
                st.session_state["agents"] = {
                    "black": agent_black,
                    "white": agent_white,
                    "config": config_obj,
                }
                st.session_state["active_checkpoint"] = str(ckpt_path)
                st.success(f"체크포인트를 불러왔습니다: {ckpt_path.name}")


def main():
    init_session_state()
    tabs = st.tabs(["학습", "플레이", "체크포인트"])

    with tabs[0]:
        run_training_ui()
    with tabs[1]:
        run_play_ui()
    with tabs[2]:
        run_checkpoint_ui()


if __name__ == "__main__":
    main()
