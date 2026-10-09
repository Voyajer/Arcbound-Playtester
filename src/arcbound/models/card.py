"""Card information model matching Java CardInfo."""

from pydantic import BaseModel, ConfigDict


class CardInfo(BaseModel):
    """Card information visible on the board. Matches Java CardInfo."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    name: str
    oracle_name: str | None = None
    mana_cost: str | None = None
    converted_mana_cost: int | None = None
    colors: list[str] = []
    color_identity: list[str] = []
    types: list[str] = []
    supertypes: list[str] = []
    subtypes: list[str] = []
    text: str | None = None
    power: int | None = None
    toughness: int | None = None
    loyalty: str | None = None
    damage: int = 0
    counters: dict[str, int] = {}
    tapped: bool = False
    attacking: bool = False
    blocking: bool = False
    blocked: bool = False
    flipped: bool = False
    face_down: bool = False
    owner: str | None = None
    controller: str | None = None
    zone: str | None = None
    abilities: list[str] = []
    keywords: list[str] = []
    revealed_to: list[str] = []
    count: int = 1
