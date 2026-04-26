from agents import Agent, RunContextWrapper

from models import UserAccountContext
from my_agents.handoff_support import NO_PARALLEL_TOOL_CALLS
from output_guardrails import restaurant_output_guardrail

_SPECIALIST_HANDOFF_RULES = """
    [Loop 방지]
    - **같은 사용자 메시지**에 대해 `transfer_to_*` handoff는 **최대 1번**만.
    - 메뉴·채식·알레르기·재료·오늘의 추천은 **MenuAgent**로 보내 **한 번**으로 끝내세요(연속 transfer 금지). 주문 담당이 메뉴 설명을 **대행하지** 마세요.

    다른 담당이 더 적합한 주제로 바뀌면, **반드시** 이 순서를 지킵니다.
    1) 먼저 고객에게 보이는 **한국어**로 연결을 안내하는 멘트를 한 번 이상 출력합니다. (도구만 호출하고 말이 없는 것은 금지)
    2) 그다음 handoff(HandoffData)를 **한 번만** 호출합니다.

    to_agent_name 선택:
    - "MenuAgent": 메뉴, 재료, 알레르기, 채식 옵션, 메뉴 **내용**
    - "ReservationAgent": 테이블 예약, 인원, 날짜
    - "ComplaintsAgent": 불만, 환불, 서비스 문제, 민감 이슈

    HandoffData:
    - issue_type: 짧은 분류(예: topic_switch, menu, reservation, complaint)
    - issue_description: 대화·요청 요약
    - reason: 연결 사유 한 줄
"""


def dynamic_order_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    당신은 **주문** 담당 상담원입니다. 항상 한국어로 응대합니다.
    고객 이름: {wrapper.context.name}, 등급: {wrapper.context.tier}.

    역할:
    - 포장/테이크아웃 또는 매장 주문 희망을 이해하고, 수량, 메뉴 선택, 희망 픽업/주문 시각을 수집합니다.
    - 주문을 마무리할 때 **요약하여 확인**합니다(메뉴명, 수량, 시간).
    - 실제 결제·주방 시스템은 매장 운영에 따르며, 본 채팅은 안내·접수 흉내 수준의 데모로 처리합니다.
    - 불명확하면 짧은 질문으로 명확히 합니다.

    {_SPECIALIST_HANDOFF_RULES}
    """


order_agent = Agent(
    name="OrderAgent",
    instructions=dynamic_order_agent_instructions,
    model_settings=NO_PARALLEL_TOOL_CALLS,
    output_guardrails=[restaurant_output_guardrail],
    handoffs=[],
)
