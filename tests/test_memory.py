"""Phase II/III: context-window management and the session store."""

from __future__ import annotations

import time

import pytest

from app.conversation.memory import estimate_messages_tokens, plan_window, render_transcript
from app.conversation.session import Session, SessionStore, Turn
from app.conversation.tokens import estimate_tokens
from app.llm.base import ChatMessage
from tests.conftest import make_turn


# -- token estimation -------------------------------------------------------

def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello there")
    long = estimate_tokens("hello there " * 20)
    assert 0 < short < long


def test_estimate_messages_includes_per_message_overhead() -> None:
    one = estimate_messages_tokens([ChatMessage("user", "abc")])
    two = estimate_messages_tokens([ChatMessage("user", "abc"), ChatMessage("assistant", "abc")])
    assert two > 2 * estimate_tokens("abc")
    assert two - one >= 4


# -- window planning --------------------------------------------------------

def test_empty_history_plans_nothing() -> None:
    plan = plan_window([], 1000)
    assert plan.kept == [] and plan.evicted == [] and not plan.evicted_any


def test_everything_kept_when_it_fits() -> None:
    turns = [make_turn(f"q{i}") for i in range(4)]
    plan = plan_window(turns, 10_000)
    assert plan.kept == turns
    assert not plan.evicted_any


def test_oldest_turns_are_evicted_first() -> None:
    turns = [make_turn(f"q{i}", chars=400) for i in range(10)]
    plan = plan_window(turns, 300, min_verbatim_turns=1)
    assert plan.evicted_any
    # The partition is order-preserving and contiguous.
    assert plan.evicted + plan.kept == turns
    # The newest turn always survives.
    assert plan.kept[-1] is turns[-1]


def test_kept_tokens_respect_budget_above_the_floor() -> None:
    turns = [make_turn(f"q{i}", chars=200) for i in range(12)]
    budget = 400
    plan = plan_window(turns, budget, min_verbatim_turns=1)
    # Only the guaranteed floor turn may push past the budget.
    overshoot = plan.kept_tokens - plan.kept[0].token_cost()
    assert overshoot <= budget


def test_min_verbatim_floor_beats_the_budget() -> None:
    """The turn being replied to must never be dropped to satisfy a budget."""
    turns = [make_turn(f"q{i}", chars=5000) for i in range(4)]
    plan = plan_window(turns, budget_tokens=10, min_verbatim_turns=2)
    assert len(plan.kept) == 2
    assert plan.kept_tokens > 10


def test_single_oversized_turn_is_kept_not_dropped() -> None:
    turns = [make_turn("huge", chars=20_000)]
    plan = plan_window(turns, budget_tokens=5, min_verbatim_turns=1)
    assert plan.kept == turns and not plan.evicted


def test_render_transcript_labels_both_speakers() -> None:
    text = render_transcript([Turn(user="where is it", assistant="checking")])
    assert "Customer: where is it" in text
    assert "Agent: checking" in text


def test_render_transcript_skips_empty_assistant() -> None:
    assert "Agent:" not in render_transcript([Turn(user="hi", assistant="")])


# -- session store ----------------------------------------------------------

async def test_get_or_create_returns_same_session(store: SessionStore) -> None:
    first = await store.get_or_create(None)
    again = await store.get_or_create(first.session_id)
    assert again is first


async def test_unknown_id_from_client_is_honoured(store: SessionStore) -> None:
    session = await store.get_or_create("s_from_a_refreshed_tab")
    assert session.session_id == "s_from_a_refreshed_tab"


async def test_reset_clears_history_but_keeps_id(store: SessionStore) -> None:
    session = await store.get_or_create(None)
    session.turns.append(make_turn("hello"))
    session.facts["order_id"] = "NIM-11112222"
    fresh = await store.reset(session.session_id)
    assert fresh.session_id == session.session_id
    assert fresh.turns == [] and fresh.facts == {}


async def test_expired_sessions_are_dropped(store: SessionStore) -> None:
    session = await store.get_or_create(None)
    session.last_active = time.time() - 10_000
    assert await store.get(session.session_id) is None


async def test_store_is_capacity_bounded(store: SessionStore) -> None:
    """A client-supplied id must not be able to grow the store without bound."""
    for index in range(12):
        created = await store.get_or_create(f"s_{index}")
        # Distinct timestamps so LRU eviction is deterministic.
        created.last_active = time.time() + index
    stats = await store.stats()
    assert stats["active_sessions"] <= store.max_sessions


async def test_delete_removes_session(store: SessionStore) -> None:
    session = await store.get_or_create(None)
    assert await store.delete(session.session_id) is True
    assert await store.delete(session.session_id) is False


def test_turn_token_cost_grows_with_content() -> None:
    small = Turn(user="hi", assistant="ok")
    big = Turn(user="hi" * 500, assistant="ok" * 500)
    assert big.token_cost() > small.token_cost()


def test_session_serialises_for_the_api() -> None:
    session = Session(session_id="s_x")
    session.turns.append(make_turn("hello"))
    data = session.as_dict()
    assert data["session_id"] == "s_x"
    assert data["turn_count"] == 1
    assert data["turns"][0]["user"] == "hello"
