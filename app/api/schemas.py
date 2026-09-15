"""Wire format for the REST and WebSocket APIs.

The WebSocket protocol is deliberately one JSON object per line in each
direction, with a mandatory ``type`` discriminator. Both directions are
validated: an inbound frame that does not parse, or parses but does not match a
known shape, produces an ``error`` frame and leaves the socket open.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.config import settings

# --------------------------------------------------------------------------
# Inbound (client -> server)
# --------------------------------------------------------------------------

ClientFrameType = Literal["chat", "reset", "ping"]


class ClientFrame(BaseModel):
    """Any frame the browser may send."""

    type: ClientFrameType
    message: str | None = None
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def _cap_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) > settings.max_message_chars:
            raise ValueError(
                f"message exceeds {settings.max_message_chars} characters"
            )
        return value


# --------------------------------------------------------------------------
# Outbound (server -> client)
# --------------------------------------------------------------------------


class ServerFrame(BaseModel):
    """Base for every outbound frame."""

    type: str


class ReadyFrame(ServerFrame):
    type: Literal["ready"] = "ready"
    session_id: str
    stage: str
    engine: str
    model: str
    turn_count: int = 0


class StartFrame(ServerFrame):
    type: Literal["start"] = "start"
    stage: str
    intent: str
    #: Non-empty when the deterministic guard answered instead of the model.
    guarded: str = ""


class TokenFrame(ServerFrame):
    type: Literal["token"] = "token"
    text: str


class DoneFrame(ServerFrame):
    type: Literal["done"] = "done"
    stage: str
    intent: str
    guarded: str = ""
    #: Non-empty when output verification replaced the reply.
    corrected: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class CorrectionFrame(ServerFrame):
    """Replace the message already rendered for this turn.

    Emitted when output verification caught the assistant claiming live order
    visibility. The client has already streamed the offending text, so it is
    told to swap the whole message rather than append to it.
    """

    type: Literal["correction"] = "correction"
    text: str
    reason: str = "fabricated_order_data"


class ContextFrame(ServerFrame):
    type: Literal["context"] = "context"
    context: dict[str, Any] = Field(default_factory=dict)


class ErrorFrame(ServerFrame):
    type: Literal["error"] = "error"
    code: str
    message: str
    #: False when the error ended the turn but the socket is still usable.
    fatal: bool = False


class ResetFrame(ServerFrame):
    type: Literal["reset_ok"] = "reset_ok"
    session_id: str


class PongFrame(ServerFrame):
    type: Literal["pong"] = "pong"


# --------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    session_id: str
    stage: str


class TurnView(BaseModel):
    user: str
    assistant: str
    stage: str
    intent: str
    created_at: float
    guarded: str = ""
    latency_ms: float = 0.0


class SessionView(BaseModel):
    session_id: str
    stage: str
    intent: str
    facts: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    turns: list[TurnView] = Field(default_factory=list)
    turn_count: int = 0
    evicted_turns: int = 0
    created_at: float
    last_active: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    engine: str
    model: str
    reachable: bool
    model_present: bool
    detail: str | None = None
    active_sessions: int = 0
    version: str


class StatsResponse(BaseModel):
    active_sessions: int
    max_sessions: int
    ttl_seconds: int
    total_turns: int
    in_flight_generations: int
    max_concurrent_generations: int
    context_window: int
    history_token_budget: int
