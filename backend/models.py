"""Pydantic request bodies for the public backend API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

class RunRequest(BaseModel):
    case: int = Field(ge=1, le=4)
    dataset_code: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    # Optional client-generated idempotency key. Repeating a lost turn with the
    # same ID replays the stored response instead of paying for it twice.
    turn_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
    )
