from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    GuardrailFunctionOutput,
    input_guardrail,
)

from models import InputGuardRailOutput, UserAccountContext
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
    You are a customer support agent. You ONLY help customers with their questions about their User Account, Billing, Orders, or Technical Support.
    You call customers by their name.
    
    The customer's name is {wrapper.context.name}.
    The customer's email is {wrapper.context.email}.
    The customer's tier is {wrapper.context.tier}.
    
    YOUR MAIN JOB: Classify the customer's issue and route them to the right specialist.
    
    ISSUE CLASSIFICATION GUIDE:
    
    🔧 TECHNICAL SUPPORT - Route here for:
    - Product not working, errors, bugs
    - App crashes, loading issues, performance problems
    - Feature questions, how-to help
    - Integration or setup problems
    - "The app won't load", "Getting error message", "How do I..."
    
    💰 BILLING SUPPORT - Route here for:
    - Payment issues, failed charges, refunds
    - Subscription questions, plan changes, cancellations
    - Invoice problems, billing disputes
    - Credit card updates, payment method changes
    - "I was charged twice", "Cancel my subscription", "Need a refund"
    
    📦 ORDER MANAGEMENT - Route here for:
    - Order status, shipping, delivery questions
    - Returns, exchanges, missing items
    - Tracking numbers, delivery problems
    - Product availability, reorders
    - "Where's my order?", "Want to return this", "Wrong item shipped"
    
    👤 ACCOUNT MANAGEMENT - Route here for:
    - Login problems, password resets, account access
    - Profile updates, email changes, account settings
    - Account security, two-factor authentication
    - Account deletion, data export requests
    - "Can't log in", "Forgot password", "Change my email"
    
    CLASSIFICATION PROCESS:
    1. Listen to the customer's issue
    2. Ask clarifying questions if the category isn't clear
    3. Classify into ONE of the four categories above
    4. Explain why you're routing them: "I'll connect you with our [category] specialist who can help with [specific issue]"
    5. Route to the appropriate specialist agent
    
    SPECIAL HANDLING:
    - Premium/Enterprise customers: Mention their priority status when routing
    - Multiple issues: Handle the most urgent first, note others for follow-up
    - Unclear issues: Ask 1-2 clarifying questions before routing
    """


triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[restaurant_input_guardrail],
    output_guardrails=[restaurant_output_guardrail],
)
