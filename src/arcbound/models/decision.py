"""Decision request/response models matching Java AiDecisionRequest and AiDecisionResponse."""

from pydantic import BaseModel, ConfigDict


class Constraints(BaseModel):
    """Choice constraints for a decision. Matches Java Constraints."""

    model_config = ConfigDict(populate_by_name=True)

    min_choices: int = 0
    max_choices: int = 1
    is_optional: bool = False
    allow_none: bool = False


class Option(BaseModel):
    """A single option available for the current decision. Matches Java Option."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: str
    label: str
    details: dict[str, object] | None = None


class DecisionContext(BaseModel):
    """Context for a single decision point. Matches Java DecisionContext."""

    model_config = ConfigDict(populate_by_name=True)

    type: str
    description: str
    options: list[Option] = []
    constraints: Constraints | None = None
    prompt: str | None = None


class AiDecisionRequest(BaseModel):
    """Request sent from Forge to the AI server. Matches Java AiDecisionRequest."""

    model_config = ConfigDict(populate_by_name=True)

    gameId: str
    boardState: dict[str, object]
    decisionRequest: DecisionContext


class AiDecisionResponse(BaseModel):
    """Response from the AI server to Forge. Matches Java AiDecisionResponse."""

    model_config = ConfigDict(populate_by_name=True)

    decisionId: str
    decisionType: str
    value: dict[str, object] = {}
    reasoning: str | None = None
    selectedOptionId: str | None = None
    selectedOptionIds: list[str] | None = None
