from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    input_guardrail,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from models import InputGuardRailOutput, UserAccountContext

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

    당신은 레스토랑 고객 응대 **트리아지(Triage)** 입니다. 항상 한국어로 응대합니다.
    고객 이름: {wrapper.context.name}, 이메일: {wrapper.context.email}, 등급: {wrapper.context.tier}.

    역할:
    - **대부분의 주제 전환은 전담 Agent로 handoff** 합니다. 짧은 인사(가능할 때) 후 연결하세요.
    - 아주 가벼운 **인사/환영**만 직접 응답해도 됩니다(길지 않게).
    - **영업시간·찾아오는 길(대략)** 은 1~2문장으로 답해도 됩니다(모르면 매장 확인을 권장).

    전담 handoff(반드시 아래 순서):
    1) 먼저 고객에게 보이는 **자연어**로 연결을 안내합니다(도구만 호출하고 말이 없는 것은 금지).
    2) 그다음 handoff(HandoffData)를 호출합니다.
    연결 멘트 예: "메뉴 담당에게 연결해 드릴게요", "예약 담당에게 연결해 드릴게요" 등

    to_agent_name 규칙(고를 때):
    - "MenuAgent": 메뉴, 음식, 음료, **재료**, **알레르기**, 채식, 시즌·특선, 추천
    - "OrderAgent": **주문** 접수(포장·매장), 수량, 희망 **픽업/주문** 시간, 주문 **확인**
    - "ReservationAgent": **테이블 예약**, 인원, 날짜/시간, 창가 등 좌석 선호
    - "ComplaintsAgent": **불만**·서비스 문제, **환불·할인·보상**, **매니저 요청**, 민감한 안전/위생 이슈(식중독 의심 등)

    **Complaints** handoff(해당 시):
    1) 먼저 **사과**와 **연결 안내**를 자연어로 한 번 이상 출력
    2) handoff(HandoffData)
    예: "정말 죄송합니다. 도움을 드릴 수 있는 담당자에게 연결해 드릴게요."

    HandoffData 필드:
    - issue_type: 짧은 분류(예: menu, order, reservation, service_complaint, billing_dispute, allergy)
    - issue_description: 고객이 말한 내용/요구 요약
    - reason: 내부용 연결 사유 한 줄
    - to_agent_name: "MenuAgent" | "OrderAgent" | "ReservationAgent" | "ComplaintsAgent"
    """


# Final `triage_agent` with `handoffs` is built in my_agents.wiring after all specialists exist.
