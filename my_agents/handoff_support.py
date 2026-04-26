import streamlit as st
from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    handoff,
)
from agents.extensions import handoff_filters

from models import HandoffData, UserAccountContext

# Prevents parallel `transfer_to_*` calls in one model response (avoids handoff spam / max_turns).
NO_PARALLEL_TOOL_CALLS = ModelSettings(parallel_tool_calls=False)

HANDOFF_USER_MESSAGE_KEY = "handoff_user_message"
HANDOFF_TARGET_NAME_KEY = "handoff_target_name"
HANDOFF_PATH_KEY = "handoff_path"

_HANDOFF_TO_KOREAN = {
    "MenuAgent": "메뉴 전문가에게 연결합니다...",
    "OrderAgent": "주문 담당에게 연결합니다...",
    "ReservationAgent": "예약 담당에게 연결합니다...",
    "ComplaintsAgent": "고객 지원(불만) 담당에게 연결해 드릴게요...",
}


def handoff_user_message_for_target(to_agent_name: str) -> str:
    return _HANDOFF_TO_KOREAN.get(
        to_agent_name, "전문 담당에게 연결합니다..."
    )


def reset_handoff_path(current_agent_name: str) -> None:
    st.session_state[HANDOFF_PATH_KEY] = [current_agent_name]


def _handoff_target_name(handoff_item) -> str | None:
    return getattr(handoff_item, "agent_name", None)


def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
    target_agent: Agent[UserAccountContext],
):
    path = st.session_state.setdefault(HANDOFF_PATH_KEY, [])
    target_agent_name = target_agent.name

    if target_agent_name not in path:
        path.append(target_agent_name)

    visited_agents = set(path)
    target_agent.handoffs = [
        h
        for h in target_agent.handoffs
        if _handoff_target_name(h) not in visited_agents
    ]
    st.session_state[HANDOFF_TARGET_NAME_KEY] = target_agent_name
    st.session_state[HANDOFF_USER_MESSAGE_KEY] = handoff_user_message_for_target(
        target_agent_name
    )
    with st.sidebar:
        st.write(
            f"""
**Handoff** → {target_agent_name}

- **Reason:** {input_data.reason}
- **Request type:** {input_data.issue_type}
- **Context:** {input_data.issue_description}
"""
        )


def make_handoff(agent: Agent[UserAccountContext]):
    def on_handoff(
        wrapper: RunContextWrapper[UserAccountContext],
        input_data: HandoffData,
    ):
        handle_handoff(wrapper, input_data, agent)

    return handoff(
        agent=agent,
        on_handoff=on_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )
