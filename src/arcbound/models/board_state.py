"""Board state models matching Java BoardState and PlayerState."""

from pydantic import BaseModel, ConfigDict

from arcbound.models.card import CardInfo


class PlayerState(BaseModel):
    """Player-specific state visible to the focal player. Matches Java PlayerState."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    life: int = 0
    starting_life: int = 20
    poison_counters: int = 0
    mana_pool: dict[str, int] | None = None
    commanders: list[CardInfo] = []
    commander_damage: dict[str, int] = {}
    commander_tax: dict[str, int] = {}
    battlefield: list[CardInfo] = []
    hand: list[CardInfo] = []
    graveyard: list[CardInfo] = []
    exile: list[CardInfo] = []
    command_zone: list[CardInfo] = []
    library_size: int = 0
    sideboard_size: int | None = None
    hand_hidden: bool = False
    is_focal: bool = False
    opponent_id: str | None = None


class BoardState(BaseModel):
    """Complete board state for the external AI. Matches Java BoardState."""

    model_config = ConfigDict(populate_by_name=True)

    game_id: str
    active_player: str
    turn: int
    phase: str
    focal_player: str
    players: list[PlayerState] = []
    stack: list[CardInfo] = []
