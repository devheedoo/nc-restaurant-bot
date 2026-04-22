import asyncio

import dotenv
import streamlit as st
from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from openai import OpenAI

from models import InputGuardRailOutput, UserAccountContext
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


async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))


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


async def run_agent(message):

    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""

        st.session_state["text_placeholder"] = text_placeholder

        try:
            stream = Runner.run_streamed(
                triage_agent,
                message,
                session=session,
                context=user_account_ctx,
            )

            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))
        except InputGuardrailTripwireTriggered as e:
            text_placeholder.write(_input_guardrail_user_message(e))


message = st.chat_input(
    "Write a message for your assistant",
)

if message:
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    if message:
        with st.chat_message("human"):
            st.write(message)
        asyncio.run(run_agent(message))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
