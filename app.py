"""Streamlit app for training and playing an Omok reinforcement learning agent."""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import torch

from omok.agent import PolicyAgent
from omok.checkpoint import CHECKPOINT_DIR, list_checkpoints, load_checkpoint, save_checkpoint
from omok.game import OmokState
from omok.training import TrainingConfig, TrainingMetrics, train_self_play

st.set_page_config(page_title="Omok RL Trainer", layout="wide")


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
        episodes = st.number_input("에피소드 수", min_value=10, max_value=2000, value=200, step=10)
        learning_rate = st.number_input("학습률", min_value=1e-4, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")
        epsilon = st.slider("탐험 비율 (epsilon)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        report_interval = st.number_input("리포트 주기", min_value=1, max_value=200, value=20)
        rolling_window = st.number_input("승률 이동 평균 길이", min_value=5, max_value=200, value=50)
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
        placeholder_chart = st.empty()
        placeholder_metrics = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_reports = max(1, math.ceil(config.episodes / config.report_interval))
        reports_seen = 0

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
    for row in range(size):
        cols = st.columns(size)
        for col in range(size):
            cell_value = state.board[row, col]
            label = "·"
            if cell_value == 1:
                label = "●"
            elif cell_value == -1:
                label = "○"
            disabled = cell_value != 0 or state.winner is not None or state.current_player != user_turn
            if cols[col].button(label, key=f"cell_{row}_{col}", use_container_width=True, disabled=disabled):
                state.apply_move((row, col))
                if state.winner is not None:
                    if state.winner == 0:
                        st.session_state["play_message"] = "무승부입니다."
                    elif (state.winner == 1 and user_color == "black") or (state.winner == -1 and user_color == "white"):
                        st.session_state["play_message"] = "축하합니다! 당신이 승리했습니다."
                    else:
                        st.session_state["play_message"] = "아쉽네요! 에이전트가 승리했습니다."
                else:
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
                return  # ensure single move per click


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
