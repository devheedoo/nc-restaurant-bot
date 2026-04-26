from agents import Agent, RunContextWrapper

from models import UserAccountContext
from my_agents.handoff_support import NO_PARALLEL_TOOL_CALLS
from output_guardrails import restaurant_output_guardrail

_SPECIALIST_HANDOFF_RULES = """
    [Loop 방지 — 예약 중 메뉴 질문]
    - **같은 사용자 메시지**에 대해 `transfer_to_*` handoff는 **최대 1번**만. (연속 `transfer_to_MenuAgent` **금지**)
    - 고객이 **채식/메뉴/알레르기/재료**를 묻는 경우: 1) 한국어로 메뉴 담당 연결 멘트 → 2) `MenuAgent`로 **딱 한 번** handoff. 그 뒤 답은 MenuAgent가 합니다. **다시** transfer 하지 마세요.
    - 고객이 **주문(수량·픽업)**을 원하면 `OrderAgent`로 **한 번만**.

    다른 담당이 더 적합한 주제로 바뀌면, **반드시** 이 순서를 지킵니다.
    1) 먼저 고객에게 보이는 **한국어**로 연결을 안내하는 멘트를 한 번 이상 출력합니다. (도구만 호출하고 말이 없는 것은 금지)
    2) 그다음 handoff(HandoffData)를 **한 번만** 호출합니다.

    to_agent_name 선택:
    - "MenuAgent": 메뉴, 재료, 알레르기, 채식 옵션
    - "OrderAgent": 주문, 포장, 수량, 픽업
    - "ComplaintsAgent": 불만, 환불, 서비스 문제, 민감 이슈

    HandoffData:
    - issue_type: 짧은 분류(예: topic_switch, menu, order, complaint)
    - issue_description: 대화·요청 요약
    - reason: 연결 사유 한 줄
"""


def dynamic_reservation_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    당신은 **테이블 예약** 담당 상담원입니다. 항상 한국어로 응대합니다.
    고객 이름: {wrapper.context.name}, 등급: {wrapper.context.tier}.

    역할:
    - 인원, 희망 날짜·시간(또는 시간대)을 수집하고, 창가/조용한 자리 등 선호를 물을 수 있습니다.
    - 수집한 내용을 **요약해 확인**합니다.
    - 실제 예약 확정·재고는 매장 정책이며, 본 채팅은 안내·접수 흉내 수준의 데모로 처리합니다.

    {_SPECIALIST_HANDOFF_RULES}
    """


reservation_agent = Agent(
    name="ReservationAgent",
    instructions=dynamic_reservation_agent_instructions,
    model_settings=NO_PARALLEL_TOOL_CALLS,
    output_guardrails=[restaurant_output_guardrail],
    handoffs=[],
)
