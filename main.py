import asyncio

import dotenv
import streamlit as st
from agents import (
    AgentUpdatedStreamEvent,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
    RunItemStreamEvent,
    Runner,
    SQLiteSession,
)
from openai import OpenAI

from models import (
    InputGuardRailOutput,
    RestaurantOutputGuardRailOutput,
    UserAccountContext,
)
from my_agents.handoff_support import (
    HANDOFF_TARGET_NAME_KEY,
    HANDOFF_USER_MESSAGE_KEY,
)
from my_agents.wiring import triage_agent

dotenv.load_dotenv()

client = OpenAI()

user_account_ctx = UserAccountContext(
    customer_id=1,
    name="heedo",
    email="devheedo@gmail.com",
    tier="basic",
)


if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent


def _label_for_agent_name(aname: str) -> str:
    if not aname:
        return "Assistant"
    if "Triage" in aname:
        return "Triage"
    if "Menu" in aname:
        return "Menu"
    if "Order" in aname:
        return "Order"
    if "Reservation" in aname:
        return "Reservation"
    if "Complaint" in aname:
        return "Complaints"
    return aname.replace(" Agent", "").split()[0]


def _escape_streamlit_markdown(text: str) -> str:
    return text.replace("$", "\\$")


def _join_chat_blocks(parts: list[str]) -> str:
    """Streamlit st.write는 Markdown이라 단일 \\n은 줄바꿈이 아니라 공백처럼 붙습니다."""
    return "\n\n".join(p for p in parts if p)


async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(_escape_streamlit_markdown(f"User: {message['content']}"))
                else:
                    if message["type"] == "message":
                        st.write(
                            _escape_streamlit_markdown(message["content"][0]["text"])
                        )


asyncio.run(paint_history())


def _input_guardrail_user_message(exc: InputGuardrailTripwireTriggered) -> str:
    info = exc.guardrail_result.output.output_info
    if isinstance(info, InputGuardRailOutput) and info.contains_inappropriate_language:
        return (
            "부적절한 표현이 포함된 메시지는 도와드리기 어려워요. "
            "레스토랑 이용·메뉴·예약 등 정중한 문의로 다시 말씀해 주세요."
        )
    return (
        "저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요. "
        "메뉴를 확인하거나, 예약하거나, 음식을 주문할 수 있어요."
    )


def _output_guardrail_user_message(exc: OutputGuardrailTripwireTriggered) -> str:
    info = exc.guardrail_result.output.output_info
    if isinstance(info, RestaurantOutputGuardRailOutput):
        if info.leaks_internal_information:
            return (
                "죄송합니다. 안내 과정에서 문제가 있었습니다. "
                "레스토랑 이용 문의로 다시 말씀해 주시면 정중하게 도와드릴게요."
            )
        if info.is_unprofessional_or_discourteous:
            return (
                "불편을 드려 죄송합니다. 더 정중하게 안내드리겠습니다. "
                "문의 내용을 한 번 더 말씀해 주시겠어요?"
            )
    return (
        "지금은 안내를 완료하기 어렵습니다. "
        "잠시 후 다시 말씀해 주시거나 다른 표현으로 문의해 주세요."
    )


def _flush_stream_segment(
    committed_lines: list[str],
    current_label: str | None,
    segment_body: str,
) -> None:
    text = segment_body.strip()
    if not text:
        return
    if current_label:
        committed_lines.append(f"{current_label}: {segment_body}")
    else:
        committed_lines.append(segment_body)


async def run_agent(message):

    with st.chat_message("ai"):
        text_placeholder = st.empty()

        st.session_state["text_placeholder"] = text_placeholder

        try:
            st.session_state.pop(HANDOFF_USER_MESSAGE_KEY, None)
            st.session_state.pop(HANDOFF_TARGET_NAME_KEY, None)

            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=user_account_ctx,
                max_turns=25,
            )

            committed_lines: list[str] = []
            current_label: str | None = None
            segment_body = ""

            async for event in stream.stream_events():
                if isinstance(event, AgentUpdatedStreamEvent):
                    _flush_stream_segment(committed_lines, current_label, segment_body)
                    segment_body = ""
                    aname = event.new_agent.name or ""
                    current_label = _label_for_agent_name(aname)
                elif (
                    isinstance(event, RunItemStreamEvent)
                    and event.name == "handoff_occured"
                ):
                    korean = (
                        st.session_state.pop(HANDOFF_USER_MESSAGE_KEY, None)
                        or "담당자에게 연결해 드릴게요"
                    )
                    target = st.session_state.pop(HANDOFF_TARGET_NAME_KEY, None)
                    # handoff 직전에 본문이 전혀 없을 때(도구만 호출 등) 연결 멘트 보강
                    if not segment_body.strip():
                        who = current_label or "Triage"
                        committed_lines.append(f"{who}: {korean}")
                    else:
                        _flush_stream_segment(
                            committed_lines, current_label, segment_body
                        )
                    segment_body = ""
                    current_label = None
                    suffix = f"  ([{target}])" if target else ""
                    committed_lines.append(f"Handoff: {korean}{suffix}")
                    text_placeholder.write(
                        _escape_streamlit_markdown(_join_chat_blocks(committed_lines))
                    )
                elif event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        segment_body += event.data.delta
                        parts = list(committed_lines)
                        if current_label and segment_body:
                            parts.append(f"{current_label}: {segment_body}")
                        elif segment_body:
                            parts.append(segment_body)
                        text_placeholder.write(
                            _escape_streamlit_markdown(_join_chat_blocks(parts))
                        )
            _flush_stream_segment(committed_lines, current_label, segment_body)
            text_placeholder.write(
                _escape_streamlit_markdown(
                    _join_chat_blocks(committed_lines),
                )
            )
            st.session_state["agent"] = stream.last_agent
        except InputGuardrailTripwireTriggered as e:
            text_placeholder.write(
                _escape_streamlit_markdown(
                    _join_chat_blocks(
                        [
                            "Bot: [input guardrail 작동]",
                            f"Bot: {_input_guardrail_user_message(e)}",
                        ]
                    )
                )
            )
        except OutputGuardrailTripwireTriggered as e:
            text_placeholder.write(
                _escape_streamlit_markdown(
                    _join_chat_blocks(
                        [
                            "Bot: [output guardrail 작동]",
                            f"Bot: {_output_guardrail_user_message(e)}",
                        ]
                    )
                )
            )
        except MaxTurnsExceeded as e:
            text_placeholder.write(
                _escape_streamlit_markdown(
                    _join_chat_blocks(
                        [
                            "Bot: [처리 제한에 도달했습니다]",
                            f"Bot: {e} "
                            "— 대화를 잠시 정리한 뒤, 한 가지씩 짧게 다시 말씀해 주세요. "
                            "필요하면 사이드바 **Reset memory**로 재시도할 수 있어요.",
                        ]
                    )
                )
            )


message = st.chat_input(
    "Write a message for your assistant",
)

if message:
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    if message:
        with st.chat_message("human"):
            st.write(_escape_streamlit_markdown(f"User: {message}"))
        asyncio.run(run_agent(message))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["agent"] = triage_agent
    st.write(asyncio.run(session.get_items()))
