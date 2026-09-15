"""Shared test fixtures.

Every test runs against ``MockEngine`` (``NIMBUS_ENGINE=mock``) so the suite is
fast, deterministic, and runnable on a machine that has never pulled a model.
The one thing the mock does *not* cover -- that the real model produces
in-character text -- is covered separately by ``scripts/evaluate.py``, which
needs Ollama running.
"""

from __future__ import annotations

import os

os.environ.setdefault("NIMBUS_ENGINE", "mock")

import pytest  # noqa: E402

from app.conversation.manager import ConversationManager  # noqa: E402
from app.conversation.session import Session, SessionStore, Turn  # noqa: E402
from app.domain.policy import Intent, Stage  # noqa: E402
from app.llm.mock_engine import MockEngine  # noqa: E402


@pytest.fixture
def engine() -> MockEngine:
    return MockEngine(token_delay=0.0, first_token_delay=0.0)


@pytest.fixture
def manager(engine: MockEngine) -> ConversationManager:
    return ConversationManager(engine)


@pytest.fixture
def session() -> Session:
    return Session(session_id="s_test")


@pytest.fixture
def store() -> SessionStore:
    return SessionStore(ttl_seconds=60, max_sessions=4)


def make_turn(user: str, assistant: str = "ok", chars: int = 0) -> Turn:
    """Build a turn, optionally padded to a target size for budget tests."""
    if chars:
        assistant = assistant + "x" * chars
    return Turn(user=user, assistant=assistant, stage=Stage.INTENT_TRIAGE, intent=Intent.UNKNOWN)


async def drain(manager: ConversationManager, session: Session, message: object):
    """Run a turn to completion and return (events, assistant_text)."""
    events = []
    text: list[str] = []
    async for event in manager.stream_turn(session, message):
        events.append(event)
        if event.type == "token":
            text.append(event.text)
    return events, "".join(text)


def last_chat_prompt(engine: MockEngine):
    """The most recent *chat* prompt handed to the engine.

    The rolling summariser also goes through ``stream_chat``, so ``calls[-1]``
    is often the summariser prompt rather than the turn prompt. Chat prompts
    are the ones carrying the Nimbus persona.
    """
    for call in reversed(engine.calls):
        if call and call[0].role == "system" and "You are Ava" in call[0].content:
            return call
    raise AssertionError("no chat prompt was recorded")
