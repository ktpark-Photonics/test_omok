"""Streamlit app for training and playing an Omok reinforcement learning agent."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

from omok.agent import PolicyAgent
from omok.checkpoint import CHECKPOINT_DIR, list_checkpoints, load_checkpoint, save_checkpoint
from omok.game import Move, OmokState
from omok.training import (
    AgentStanding,
    TrainingConfig,
    TrainingMetrics,
    resolve_device,
    train_self_play,
)

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
    if "play_selected_agent" not in st.session_state:
        st.session_state["play_selected_agent"] = 0
    if "active_checkpoint" not in st.session_state:
        st.session_state["active_checkpoint"] = None
    if "device_preference" not in st.session_state:
        st.session_state["device_preference"] = "auto


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


def device_options() -> Dict[str, str]:
    options = {"auto": "자동 선택 (GPU 우선)", "cpu": "CPU"}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            key = f"cuda:{index}" if torch.cuda.device_count() > 1 else "cuda"
            if key in options:
                key = f"cuda:{index}"
            options[key] = f"CUDA {index} · {name}"
    return options


def ensure_agents(config: TrainingConfig, resolved_device: Optional[torch.device] = None) -> Dict[str, object]:
    agents_bundle = st.session_state.get("agents")
    try:
        device = resolved_device or resolve_device(config.device)
    except RuntimeError as exc:
        st.error(str(exc))
        raise
    if agents_bundle and "list" not in agents_bundle:
        legacy_agents = [agents_bundle.get("black"), agents_bundle.get("white")]
        agents_bundle["list"] = [agent for agent in legacy_agents if agent is not None]
        agents_bundle.setdefault("standings", [])
    needs_new = True
    if agents_bundle:
        prev_config: TrainingConfig = agents_bundle.get("config", config)
        needs_new = (
            prev_config.board_size != config.board_size
            or prev_config.win_length != config.win_length
            or len(agents_bundle.get("list", [])) != config.num_agents
            or getattr(prev_config, "device", "auto") != config.device
        )
    if needs_new:
        agents = [
            PolicyAgent(
                board_size=config.board_size,
                learning_rate=config.learning_rate,
                epsilon=config.epsilon,
                device=device,
            )
            for _ in range(config.num_agents)
        ]
        st.session_state["agents"] = {
            "list": agents,
            "config": config,
            "standings": [],
        }
    else:
        agents_bundle["config"] = config
        agents_bundle.setdefault("standings", [])
        for agent in agents_bundle.get("list", []):
            agent.update_hyperparameters(
                learning_rate=config.learning_rate,
                epsilon=config.epsilon,
            )
            agent.to(device)
    return st.session_state["agents"]


def update_training_history(metric: TrainingMetrics, placeholder_chart, placeholder_metrics):
    st.session_state["training_history"].append(metric)
    history_records = []
    for entry in st.session_state["training_history"]:
        top = entry.standings[0] if entry.standings else None
        history_records.append(
            {
                "round": entry.round_index,
                "top_win_rate": top.win_rate if top else 0.0,
                "top_recent": top.recent_win_rate if top else 0.0,
            }
        )
    history_df = pd.DataFrame(history_records)
    if not history_df.empty:
        history_df.set_index("round", inplace=True)
        placeholder_chart.line_chart(history_df)
    else:
        placeholder_chart.empty()
    with placeholder_metrics.container():
        if metric.standings:
            top = metric.standings[0]
            runner_up = metric.standings[1] if len(metric.standings) > 1 else None
            col_a, col_b = st.columns(2)
            col_a.metric(
                "리그 1위 누적 승률",
                f"{top.win_rate:.2f}",
                delta=f"최근 {top.recent_win_rate:.2f}",
            )
            if runner_up:
                col_b.metric(
                    "리그 2위 누적 승률",
                    f"{runner_up.win_rate:.2f}",
                    delta=f"최근 {runner_up.recent_win_rate:.2f}",
                )
            else:
                col_b.metric("참여 에이전트 수", f"{len(metric.standings)}명")
        else:
            placeholder_metrics.empty()


def run_training_ui():
    st.header("🔁 리그 자기대국 학습")
    st.write(
        "여러 에이전트가 리그전을 치르며 서로 학습합니다. 라운드마다 짝을 바꿔 대국하고, 승패에 따라 보상이 적용됩니다."
    )

    with st.form("training_form"):
        board_size = st.slider("바둑판 크기", min_value=5, max_value=15, value=9)
        num_agents = st.number_input("에이전트 수 (짝수)", min_value=2, max_value=32, value=4, step=2)
        episodes = st.number_input("리그 라운드 수", min_value=1, max_value=100000, value=50, step=1)
        learning_rate = st.number_input("학습률", min_value=1e-4, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")
        epsilon = st.slider("탐험 비율 (epsilon)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        report_interval = st.number_input("리포트 주기", min_value=1, max_value=500, value=10)
        rolling_window = st.number_input("승률 이동 평균 길이", min_value=5, max_value=500, value=50)
        device_map = device_options()
        preferred_device = st.session_state.get("device_preference", "auto")
        option_values = list(device_map.keys())
        if preferred_device not in option_values:
            option_values.insert(0, preferred_device)

        def device_label(key: str) -> str:
            if key in device_map:
                return device_map[key]
            return f"사용 불가: {key}"

        device_index = option_values.index(preferred_device) if preferred_device in option_values else 0
        device_choice = st.selectbox(
            "연산 장치",
            options=option_values,
            index=device_index,
            format_func=device_label,
            help="GPU(CUDA)가 감지되면 자동으로 목록에 표시됩니다. RTX 50 시리즈 등 최신 GPU도 CUDA 빌드 PyTorch로 사용 가능합니다.",
        )
        show_training_visual = st.checkbox("학습 대국 실시간 보기", value=True)
        visual_delay = st.slider("수 시각화 지연 (초)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        submitted = st.form_submit_button("학습 시작")

    if submitted:
        if int(num_agents) % 2 != 0:
            st.error("에이전트 수는 짝수여야 합니다.")
            return
        config = TrainingConfig(
            board_size=int(board_size),
            win_length=5,
            episodes=int(episodes),
            learning_rate=float(learning_rate),
            epsilon=float(epsilon),
            report_interval=int(report_interval),
            rolling_window=int(rolling_window),
            num_agents=int(num_agents),
            device=device_choice,
        )
        st.session_state["device_preference"] = device_choice
        try:
            resolved_device = resolve_device(config.device)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        try:
            agents_bundle = ensure_agents(config, resolved_device)
        except RuntimeError:
            return
        if not show_training_visual:
            visual_delay = 0.0

        num_matches = max(1, config.num_agents // 2)

        if show_training_visual:
            col_visual, col_metrics = st.columns([1.3, 1.0])
            board_placeholders: List = []
            move_placeholders: List = []
            with col_visual:
                st.caption("실시간 리그 경기")
                cols_per_row = min(4, num_matches)
                rows = math.ceil(num_matches / cols_per_row)
                match_index = 0
                for _ in range(rows):
                    row_columns = st.columns(cols_per_row)
                    for col in row_columns:
                        if match_index >= num_matches:
                            break
                        with col:
                            st.markdown(f"**매치 {match_index + 1}**")
                            board_ph = st.empty()
                            move_ph = st.empty()
                            board_ph.markdown(
                                board_preview_html(
                                    OmokState(size=config.board_size, win_length=config.win_length),
                                    caption="대국 대기중",
                                ),
                                unsafe_allow_html=True,
                            )
                            move_ph.caption("준비 중...")
                            board_placeholders.append(board_ph)
                            move_placeholders.append(move_ph)
                            match_index += 1
        else:
            col_metrics = st.container()
            board_placeholders = []
            move_placeholders = []

        with col_metrics:
            placeholder_chart = st.empty()
            placeholder_metrics = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_reports = max(1, math.ceil(config.episodes / config.report_interval))
        reports_seen = 0

        def visualize_move(
            round_idx: int,
            match_idx: int,
            move_idx: int,
            move: Optional[Move],
            player: int,
            snapshot: OmokState,
            pairing: Tuple[int, int],
        ) -> None:
            if match_idx >= len(board_placeholders):
                return
            black_idx, white_idx = pairing
            title = (
                f"라운드 {round_idx} · 에이전트 {black_idx + 1} (흑) vs 에이전트 {white_idx + 1} (백)"
            )
            if move_idx == -1:
                board_placeholders[match_idx].markdown(
                    board_preview_html(snapshot, caption=title),
                    unsafe_allow_html=True,
                )
                move_placeholders[match_idx].markdown("대국 시작!", unsafe_allow_html=True)
                return
            if move_idx == -2:
                result_text = "무승부"
                if player == 1:
                    result_text = "흑 승리"
                elif player == -1:
                    result_text = "백 승리"
                board_placeholders[match_idx].markdown(
                    board_preview_html(snapshot, caption=title),
                    unsafe_allow_html=True,
                )
                move_placeholders[match_idx].markdown(f"결과: {result_text}", unsafe_allow_html=True)
                return
            color = "흑" if player == 1 else "백"
            board_placeholders[match_idx].markdown(
                board_preview_html(snapshot, highlight=move, caption=title),
                unsafe_allow_html=True,
            )
            if move is not None:
                move_placeholders[match_idx].markdown(
                    f"{color} ({move[0] + 1}, {move[1] + 1})",
                    unsafe_allow_html=True,
                )
            if visual_delay > 0 and move_idx >= 0:
                time.sleep(visual_delay)

        def progress_callback(metric: TrainingMetrics):
            nonlocal reports_seen
            reports_seen += 1
            update_training_history(metric, placeholder_chart, placeholder_metrics)
            progress_bar.progress(min(1.0, reports_seen / total_reports))
            if metric.standings:
                leader = metric.standings[0]
                status_text.info(
                    f"라운드 {metric.round_index} / {config.episodes} · 1위 에이전트 {leader.agent_index + 1} 승률 {leader.win_rate:.2f}"
                )
            else:
                status_text.info(f"라운드 {metric.round_index} 진행 중")

        start_time = time.time()
        result = train_self_play(
            config,
            agents=agents_bundle.get("list"),
            progress_callback=progress_callback,
            move_callback=visualize_move if show_training_visual else None,
            device=resolved_device,
        )
        progress_bar.progress(1.0)
        elapsed = time.time() - start_time
        st.success(f"학습이 완료되었습니다! (소요 시간: {elapsed:.1f}초)")
        st.session_state["agents"] = {
            "list": result.agents,
            "config": config,
            "standings": result.standings,
        }
        st.session_state["active_checkpoint"] = None
        st.session_state["play_selected_agent"] = min(
            st.session_state.get("play_selected_agent", 0), len(result.agents) - 1
        )

        standings_df = pd.DataFrame([standing.to_dict() for standing in result.standings])
        if not standings_df.empty:
            standings_df.set_index("agent", inplace=True)
            st.subheader("🏅 최종 리그 순위")
            st.dataframe(standings_df)

    if st.session_state["training_history"]:
        st.subheader("📈 최근 학습 지표")
        history_df = pd.DataFrame(
            [
                {
                    "round": m.round_index,
                    "leader": m.standings[0].agent_index + 1 if m.standings else None,
                    "leader_win_rate": m.standings[0].win_rate if m.standings else 0.0,
                    "leader_recent": m.standings[0].recent_win_rate if m.standings else 0.0,
                }
                for m in st.session_state["training_history"]
            ]
        )
        st.dataframe(history_df.set_index("round"))


def get_agent_pool() -> List[PolicyAgent]:
    agents_bundle = st.session_state.get("agents")
    if not agents_bundle:
        return []
    return agents_bundle.get("list", [])


def get_selected_agent() -> Optional[PolicyAgent]:
    agents = get_agent_pool()
    if not agents:
        return None
    index = min(st.session_state.get("play_selected_agent", 0), len(agents) - 1)
    return agents[index]


def agent_move(state: OmokState, color: str) -> None:
    agent = get_selected_agent()
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
    agent_pool = get_agent_pool()
    standings = agents_bundle.get("standings", []) if agents_bundle else []
    if agent_pool:
        st.session_state["play_selected_agent"] = min(
            st.session_state.get("play_selected_agent", 0), len(agent_pool) - 1
        )
    else:
        st.session_state["play_selected_agent"] = 0
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

        if agent_pool:
            standings_map = {standing.agent_index: standing for standing in standings}

            def format_agent(idx: int) -> str:
                standing = standings_map.get(idx)
                if standing:
                    return f"에이전트 {idx + 1} · 승률 {standing.win_rate:.2f}"
                return f"에이전트 {idx + 1}"

            options = list(range(len(agent_pool)))
            default_index = (
                options.index(st.session_state["play_selected_agent"])
                if options and st.session_state["play_selected_agent"] in options
                else 0
            )
            selected_agent = st.selectbox(
                "대결할 에이전트",
                options=options,
                format_func=format_agent,
                index=default_index,
            )
            if selected_agent != st.session_state["play_selected_agent"]:
                st.session_state["play_selected_agent"] = selected_agent
                st.session_state["play_state"] = OmokState(size=state.size, win_length=5)
                st.session_state["play_message"] = "새로운 에이전트와의 대국을 시작합니다!"
                state = st.session_state["play_state"]
                if st.session_state["play_user_color"] == "white":
                    agent_move(state, "black")
                    if state.winner is None:
                        st.session_state["play_message"] = "당신의 차례입니다."

            selected_stats = standings_map.get(st.session_state["play_selected_agent"])
            if selected_stats:
                st.metric(
                    "선택 에이전트 최근 승률",
                    f"{selected_stats.recent_win_rate:.2f}",
                    delta=f"최고 {selected_stats.best_recent_win_rate:.2f}",
                )
                st.caption(
                    f"누적 전적: {selected_stats.wins}승 {selected_stats.draws}무 {selected_stats.losses}패"
                )
        else:
            st.info("학습을 완료하거나 체크포인트를 불러오면 에이전트를 선택할 수 있습니다.")

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

        if standings:
            leaderboard_df = pd.DataFrame([standing.to_dict() for standing in standings])
            if not leaderboard_df.empty:
                leaderboard_df.set_index("agent", inplace=True)
                st.caption("현재 리그 순위 (상위 10)")
                st.dataframe(leaderboard_df.head(10))

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
                save_checkpoint(
                    path,
                    agents_bundle.get("list", []),
                    config,
                    standings=agents_bundle.get("standings"),
                    notes=notes,
                )
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
            saved_device = config.get("device", "auto")
            saved_label = device_options().get(saved_device, saved_device)
            st.caption(f"저장 당시 장치: {saved_label}")
            if metadata.get("notes"):
                st.write(metadata["notes"])
            if st.button("이 체크포인트 불러오기", key=f"load_{ckpt_path.stem}"):
                preferred_device = st.session_state.get("device_preference", "auto")
                try:
                    load_device = resolve_device(preferred_device)
                except RuntimeError as exc:
                    st.warning(f"{exc} CPU로 불러옵니다.")
                    load_device = torch.device("cpu")
                config_data = payload.get("config", {})
                config_obj = TrainingConfig(
                    board_size=config_data.get("board_size", 9),
                    win_length=config_data.get("win_length", 5),
                    learning_rate=config_data.get("learning_rate", 1e-3),
                    epsilon=config_data.get("epsilon", 0.2),
                    num_agents=config_data.get("num_agents", len(payload.get("agents", [])) or 2),
                    device=preferred_device,
                )
                agent_states = payload.get("agents")
                agents: List[PolicyAgent] = []
                if agent_states:
                    for state_dict in agent_states:
                        agent = PolicyAgent(
                            board_size=config_obj.board_size,
                            learning_rate=config_obj.learning_rate,
                            epsilon=config_obj.epsilon,
                            device=load_device,
                        )
                        agent.load_state_dict(state_dict)
                        agents.append(agent)
                else:
                    # 호환성을 위해 기존 흑/백 에이전트 형식도 처리
                    black_state = payload.get("black")
                    white_state = payload.get("white")
                    if black_state and white_state:
                        agent_black = PolicyAgent(
                            board_size=config_obj.board_size,
                            learning_rate=config_obj.learning_rate,
                            epsilon=config_obj.epsilon,
                            device=load_device,
                        )
                        agent_white = PolicyAgent(
                            board_size=config_obj.board_size,
                            learning_rate=config_obj.learning_rate,
                            epsilon=config_obj.epsilon,
                            device=load_device,
                        )
                        agent_black.load_state_dict(black_state)
                        agent_white.load_state_dict(white_state)
                        agents = [agent_black, agent_white]
                        config_obj.num_agents = 2

                standings_payload = payload.get("standings", [])
                standings_list: List[AgentStanding] = []
                for entry in standings_payload:
                    try:
                        standings_list.append(AgentStanding(**entry))
                    except TypeError:
                        standings_list.append(
                            AgentStanding(
                                agent_index=entry.get("agent", 1) - 1,
                                wins=entry.get("wins", 0),
                                losses=entry.get("losses", 0),
                                draws=entry.get("draws", 0),
                                total_games=entry.get("games", 0),
                                win_rate=entry.get("win_rate", 0.0),
                                recent_win_rate=entry.get("recent_win_rate", 0.0),
                                best_recent_win_rate=entry.get("best_recent", 0.0),
                                average_loss=entry.get("avg_loss", 0.0),
                            )
                        )

                st.session_state["agents"] = {
                    "list": agents,
                    "config": config_obj,
                    "standings": standings_list,
                }
                st.session_state["play_state"] = OmokState(size=config_obj.board_size, win_length=config_obj.win_length)
                st.session_state["play_selected_agent"] = 0
                st.session_state["play_message"] = "체크포인트 에이전트와의 대국을 시작해보세요!"
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
