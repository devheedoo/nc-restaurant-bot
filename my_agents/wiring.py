"""Wire Triage and specialist agents with mutual handoffs (mutable `.handoffs`)."""

from agents import Agent

from models import UserAccountContext
from my_agents.complaints_agent import complaints_agent
from my_agents.handoff_support import make_handoff
from my_agents.menu_agent import menu_agent
from my_agents.order_agent import order_agent
from my_agents.reservation_agent import reservation_agent
from my_agents.triage_agent import (
    dynamic_triage_agent_instructions,
    restaurant_input_guardrail,
)
from output_guardrails import restaurant_output_guardrail

specialists: list[Agent[UserAccountContext]] = [
    menu_agent,
    order_agent,
    reservation_agent,
    complaints_agent,
]
for a in specialists:
    a.handoffs = [make_handoff(t) for t in specialists if t is not a]

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[restaurant_input_guardrail],
    output_guardrails=[restaurant_output_guardrail],
    handoffs=[make_handoff(m) for m in specialists],
)
