from pydantic import BaseModel


class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    email: str
    tier: str = "basic"  # premium entreprise


class InputGuardRailOutput(BaseModel):
    """Structured result from the input guardrail classifier agent."""

    is_off_topic: bool
    contains_inappropriate_language: bool
    reason: str
