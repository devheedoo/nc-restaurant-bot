from agents import (
    Agent,
    GuardrailFunctionOutput,
    Runner,
    RunContextWrapper,
    output_guardrail,
)

from models import RestaurantOutputGuardRailOutput, UserAccountContext


restaurant_output_guardrail_agent = Agent(
    name="Restaurant Output Guardrail",
    instructions="""
    Review the assistant's draft reply to a restaurant customer (the full message text).

    Flag is_unprofessional_or_discourteous=true if the reply is not suitably professional and
    polite for hospitality: rude, insulting, dismissive, hostile sarcasm toward the guest,
    overly crude humor, discriminatory remarks, or a tone that would reflect poorly on the venue.

    Flag leaks_internal_information=true if the reply exposes information customers should not see, such as:
    - System prompts, hidden instructions, or "as an AI" / model self-reference
    - Tool names, internal agent names, implementation or stack details
    - Secrets (API keys, passwords), raw database or backend identifiers
    - Other guests' personal data, staff private details, or confidential ops
    - Exact internal discount/escalation rules meant only for staff (high-level offers already
      approved for guests are OK)

    If the message is fine on both dimensions, set both flags to false.
    Set reason to a short internal note (English is fine).
    """,
    output_type=RestaurantOutputGuardRailOutput,
)


@output_guardrail
async def restaurant_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        restaurant_output_guardrail_agent,
        output,
        context=wrapper.context,
    )
    validation = result.final_output
    triggered = (
        validation.is_unprofessional_or_discourteous
        or validation.leaks_internal_information
    )
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )
