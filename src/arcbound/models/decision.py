"""Decision request/response models matching Java AiDecisionRequest and AiDecisionResponse."""

from pydantic import BaseModel, ConfigDict, Field


class Constraints(BaseModel):
    """Choice constraints for a decision. Matches Java Constraints.

    Java serializes these as camelCase (minChoices, maxChoices, isOptional,
    allowNone); the aliases map them onto the snake_case fields.
    """

    model_config = ConfigDict(populate_by_name=True)

    min_choices: int = Field(default=0, alias="minChoices")
    max_choices: int = Field(default=1, alias="maxChoices")
    is_optional: bool = Field(default=False, alias="isOptional")
    allow_none: bool = Field(default=False, alias="allowNone")


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
    # Number of cards tuck-returned if this mulligan is taken (London mulligan).
    # Only set for MULLIGAN decisions; None otherwise.
    cards_to_return: int | None = Field(default=None, alias="cardsToReturn")
    # How many times the focal player has already mulliganed this game.
    # Only set for MULLIGAN decisions; None otherwise.
    mulligan_count: int | None = Field(default=None, alias="mulliganCount")


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
