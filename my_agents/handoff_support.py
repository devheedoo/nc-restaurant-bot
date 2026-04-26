from typing import Literal

import streamlit as st
from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    handoff,
)
from agents.exceptions import ModelBehaviorError
from agents.extensions import handoff_filters
from pydantic import Field, create_model

from models import HandoffData, UserAccountContext

# Prevents parallel `transfer_to_*` calls in one model response (avoids handoff spam / max_turns).
NO_PARALLEL_TOOL_CALLS = ModelSettings(parallel_tool_calls=False)

HANDOFF_USER_MESSAGE_KEY = "handoff_user_message"
HANDOFF_TARGET_NAME_KEY = "handoff_target_name"
HANDOFF_PATH_KEY = "handoff_path"
HANDOFF_USED_KEY = "handoff_used"

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
    st.session_state[HANDOFF_USED_KEY] = False


def _handoff_is_enabled(target_agent_name: str):
    def is_enabled(
        wrapper: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
    ) -> bool:
        path = st.session_state.get(HANDOFF_PATH_KEY, [])
        return (
            not st.session_state.get(HANDOFF_USED_KEY, False)
            and target_agent_name not in path
        )

    return is_enabled


def _handoff_input_type_for(target_agent_name: str) -> type[HandoffData]:
    return create_model(
        f"{target_agent_name}HandoffData",
        __base__=HandoffData,
        to_agent_name=(
            Literal[target_agent_name],
            Field(
                ...,
                description=f'Must be exactly "{target_agent_name}".',
            ),
        ),
    )


def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
    target_agent: Agent[UserAccountContext],
):
    if st.session_state.get(HANDOFF_USED_KEY, False):
        raise ModelBehaviorError(
            "Only one handoff is allowed for a single user message."
        )

    path = st.session_state.setdefault(HANDOFF_PATH_KEY, [])
    target_agent_name = target_agent.name
    if input_data.to_agent_name != target_agent_name:
        raise ModelBehaviorError(
            f"Handoff payload target {input_data.to_agent_name!r} does not match "
            f"the invoked handoff target {target_agent_name!r}."
        )

    if target_agent_name not in path:
        path.append(target_agent_name)

    st.session_state[HANDOFF_USED_KEY] = True
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
        input_type=_handoff_input_type_for(agent.name),
        input_filter=handoff_filters.remove_all_tools,
        is_enabled=_handoff_is_enabled(agent.name),
    )
