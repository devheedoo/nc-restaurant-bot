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


class RestaurantOutputGuardRailOutput(BaseModel):
    """Structured result from the restaurant assistant output guardrail."""

    is_unprofessional_or_discourteous: bool
    leaks_internal_information: bool
    reason: str


class HandoffData(BaseModel):
    """Structured payload passed when handing off from triage to a specialist."""

    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
