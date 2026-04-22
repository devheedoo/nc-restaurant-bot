import streamlit as st
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    GuardrailFunctionOutput,
    handoff,
    input_guardrail,
)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from models import HandoffData, InputGuardRailOutput, UserAccountContext
from my_agents.complaints_agent import complaints_agent
from output_guardrails import restaurant_output_guardrail


input_guardrail_agent = Agent(
    name="Restaurant Input Guardrail",
    instructions="""
    You classify the user's latest message for a restaurant customer-chat assistant.

    ALLOW (not a violation) — messages that are about:
    - Menu, food, drinks, allergens, ingredients, specials
    - Reservations, wait times, seating, hours, location, parking
    - Orders (takeout/delivery/dine-in), payment at the restaurant, receipts
    - Service issues, complaints, praise, staff behavior, cleanliness
    - Policies (dress code, kids, pets), events, private dining
    - Brief greetings or light small talk at the start of a conversation

    Treat as OFF-TOPIC (is_off_topic=true) when the message is unrelated to the restaurant /
    hospitality context, for example: general knowledge, philosophy, politics unrelated to
    the venue, coding, homework, medical/legal advice, or other domains with no clear link
    to this restaurant assistant.

    Treat as INAPPROPRIATE LANGUAGE (contains_inappropriate_language=true) when the message
    includes profanity used as insults, slurs, sexual harassment, hate, or threats — even if
    the topic mentions food or the restaurant.

    If unsure whether a message is on-topic, prefer is_off_topic=false when there is any
    plausible restaurant or dining connection.

    Set reason to a short internal explanation (English is fine).
    """,
    output_type=InputGuardRailOutput,
)


@input_guardrail
async def restaurant_input_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
    input: str,
):
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=wrapper.context,
    )
    out = result.final_output
    triggered = out.is_off_topic or out.contains_inappropriate_language
    return GuardrailFunctionOutput(
        output_info=out,
        tripwire_triggered=triggered,
    )


def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    {RECOMMENDED_PROMPT_PREFIX}

    당신은 레스토랑 고객 응대 트리아지(Triage)입니다. 항상 한국어로 응대합니다.
    고객 이름: {wrapper.context.name}, 이메일: {wrapper.context.email}, 등급: {wrapper.context.tier}.

    역할:
    - 메뉴·영업시간·위치·예약·웨이팅·주문·포장 등 일반 문의에는 직접 정중하게 안내합니다.
    - 음식 품질, 서비스, 직원 태도, 청결, 대기, 과금 불만 등 불만·불쾌 경험이 분명하면
      먼저 짧게 사과한 뒤, 예시처럼 말하고 불만 전담(Complaints Agent)으로 handoff 합니다.
      예: "정말 죄송합니다. 도움을 드릴 수 있는 담당자에게 연결해 드릴게요."

    불만 handoff 기준 (해당 시 반드시 전담으로 연결):
    - 음식이 맛없었다, 직원이 불친절했다, 기다림이 길었다 등 명확한 부정 경험
    - 환불·할인·보상·매니저 요구
    - 위생·안전·알레르기·식중독 의심 등 민감 이슈

    handoff 시 HandoffData를 채웁니다:
    - issue_type: 짧은 분류 (예: service_complaint, food_quality, billing_dispute)
    - issue_description: 고객이 말한 내용 요약
    - reason: 연결 사유 한 줄
    - to_agent_name: "ComplaintsAgent"

    일반 문의만 있고 불만이 아니면 handoff 하지 말고 직접 답합니다.
    """


def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
):
    with st.sidebar:
        st.write(
            f"""
**Handoff** → {input_data.to_agent_name}

- **Reason:** {input_data.reason}
- **Issue type:** {input_data.issue_type}
- **Description:** {input_data.issue_description}
"""
        )


def make_handoff(agent: Agent[UserAccountContext]):
    return handoff(
        agent=agent,
        on_handoff=handle_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )


triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[restaurant_input_guardrail],
    output_guardrails=[restaurant_output_guardrail],
    handoffs=[make_handoff(complaints_agent)],
)
