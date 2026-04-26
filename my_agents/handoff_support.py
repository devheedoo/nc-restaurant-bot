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


def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
):
    st.session_state[HANDOFF_TARGET_NAME_KEY] = input_data.to_agent_name
    st.session_state[HANDOFF_USER_MESSAGE_KEY] = handoff_user_message_for_target(
        input_data.to_agent_name
    )
    with st.sidebar:
        st.write(
            f"""
**Handoff** → {input_data.to_agent_name}

- **Reason:** {input_data.reason}
- **Request type:** {input_data.issue_type}
- **Context:** {input_data.issue_description}
"""
        )


def make_handoff(agent: Agent[UserAccountContext]):
    return handoff(
        agent=agent,
        on_handoff=handle_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )
