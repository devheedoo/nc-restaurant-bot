import asyncio

import dotenv
import streamlit as st
from agents import (
    AgentUpdatedStreamEvent,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    RunItemStreamEvent,
    SQLiteSession,
)
from openai import OpenAI

from models import (
    InputGuardRailOutput,
    RestaurantOutputGuardRailOutput,
    UserAccountContext,
)
from my_agents.triage_agent import triage_agent

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

# Triage 답변 → handoff 안내 → Complaints 답변 순으로 표시할 때 쓰는 고정 문구
HANDOFF_CHAT_NOTICE = "[Complaints Agent로 handoff]"


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
            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=user_account_ctx,
            )

            committed_lines: list[str] = []
            current_label: str | None = None
            segment_body = ""

            async for event in stream.stream_events():
                if isinstance(event, AgentUpdatedStreamEvent):
                    _flush_stream_segment(committed_lines, current_label, segment_body)
                    segment_body = ""
                    aname = event.new_agent.name or ""
                    if "Complaint" in aname:
                        current_label = "Complaints"
                    elif aname:
                        current_label = aname.replace(" Agent", "").split()[0]
                    else:
                        current_label = "Assistant"
                elif isinstance(event, RunItemStreamEvent) and event.name == "handoff_occured":
                    # 모델이 handoff만 호출하고 트리아지 멘트를 스트리밍하지 않는 경우가 있어 보강
                    if not segment_body.strip():
                        committed_lines.append(
                            "Triage: 정말 죄송합니다. "
                            "도움을 드릴 수 있는 담당자에게 연결해 드릴게요."
                        )
                    else:
                        _flush_stream_segment(committed_lines, current_label, segment_body)
                    segment_body = ""
                    current_label = None
                    # (1) Triage (2) handoff 안내 (3) Complaints — 가운데 줄을 답변에 명시
                    committed_lines.append(f"Handoff: {HANDOFF_CHAT_NOTICE}")
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
