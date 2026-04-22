import streamlit as st
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
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
    레스토랑 고객 채팅 어시스턴트에 들어온 사용자 메시지를 분류합니다.

    허용(위반 아님) — 다음에 해당하면 is_off_topic=false, contains_inappropriate_language=false 쪽으로 둡니다:
    - 메뉴, 음식, 음료, 알레르기, 재료, 시즌·특선 메뉴
    - 예약, 웨이팅, 좌석, 영업시간, 위치, 주차
    - 주문(포장·배달·매장), 매장 결제, 영수증
    - 서비스 불만·칭찬, 직원 태도, 청결
    - 매장 규정(드레스코드, 어린이, 반려동물), 이벤트, 단체·프라이빗 다이닝
    - 대화 시작 시 짧은 인사나 가벼운 스몰토크

    주제 이탈(OFF-TOPIC): 레스토랑·외식 맥락과 무관하면 is_off_topic=true 로 둡니다. 예: 일반 상식·철학,
    매장과 무관한 정치, 코딩, 숙제, 의료·법률 자문 등 이 어시스턴트와 연결하기 어려운 영역.

    부적절한 언어: 모욕·비속어, 혐오·차별, 성희롱, 위협 등이 있으면 contains_inappropriate_language=true 로 둡니다.
    주제가 음식·레스토랑이어도 마찬가지입니다.

    애매하면 레스토랑·다이닝과 조금이라도 연결될 수 있으면 is_off_topic=false 를 선호합니다.

    reason에는 내부용으로 짧은 판단 근거를 한국어로 적습니다.
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
      불만 전담(Complaints Agent)으로 handoff 하되, **반드시 아래 순서를 지킵니다.**
      1) 먼저 고객에게 보이는 **자연어 메시지**를 반드시 한 번 이상 출력해 사과하고 연결을 안내합니다.
         (도구 호출만 하고 말을 하지 않는 것은 금지입니다.)
      2) 그 다음에 handoff(HandoffData)를 호출합니다.
      예시 문구: "정말 죄송합니다. 도움을 드릴 수 있는 담당자에게 연결해 드릴게요."

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
