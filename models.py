from pydantic import BaseModel, Field


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
    """Structured payload for handoff between Triage and specialists (or specialist ↔ specialist)."""

    to_agent_name: str = Field(
        ...,
        description='Target agent id, e.g. "MenuAgent", "OrderAgent", "ReservationAgent", "ComplaintsAgent".',
    )
    issue_type: str = Field(
        ...,
        description="Short request category, e.g. menu, order, reservation, or complaint subtypes.",
    )
    issue_description: str = Field(
        ...,
        description="User request / conversation context summary for the receiving agent.",
    )
    reason: str = Field(..., description="One-line handoff reason (internal / routing).")
