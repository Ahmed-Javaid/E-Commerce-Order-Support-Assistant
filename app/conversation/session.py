"""Session state and the in-process session store.

A *session* is one browser tab's conversation: an ordered list of completed
turns, the stage machine's current position, the identifiers the customer has
supplied, and a rolling summary of whatever has fallen out of the verbatim
window.

The store is in-process and in-memory by design. The assignment forbids
external state (no RAG, no database), and a single-process store is the honest
choice for a system that also holds a single local model in memory: sharding
sessions across workers would not help, because the model is the bottleneck.
The known-limitations section of the README states the consequence -- sessions
do not survive a restart.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field

from app.config import settings
from app.conversation.tokens import PER_MESSAGE_TOKEN_OVERHEAD, estimate_tokens
from app.domain.policy import Intent, Stage


@dataclass
class Turn:
    """One completed customer/assistant exchange."""

    user: str
    assistant: str = ""
    stage: Stage = Stage.GREETING
    intent: Intent = Intent.UNKNOWN
    created_at: float = field(default_factory=time.time)
    #: Set when the deterministic guard answered instead of the model.
    guarded: str = ""
    latency_ms: float = 0.0

    def token_cost(self) -> int:
        """Prompt cost of replaying this turn verbatim."""
        return (
            estimate_tokens(self.user)
            + estimate_tokens(self.assistant)
            + 2 * PER_MESSAGE_TOKEN_OVERHEAD
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "stage": self.stage.value,
            "intent": self.intent.value,
            "created_at": self.created_at,
            "guarded": self.guarded,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class Session:
    """Everything the conversation manager needs to answer the next turn."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    stage: Stage = Stage.GREETING
    intent: Intent = Intent.UNKNOWN
    #: Identifiers pinned out of the dialogue; survive history eviction.
    facts: dict[str, str] = field(default_factory=dict)
    #: Rolling summary of turns that have been evicted from the verbatim window.
    summary: str = ""
    #: Lane the customer was in before switching topic, so we can offer to resume.
    #: Only the lane is parked, not a snapshot of the facts: facts are merged
    #: rather than replaced, so a snapshot would never differ. The consequence
    #: is that a session tracks one order at a time -- see README limitations.
    suspended_intent: Intent | None = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    #: Serialises generations for this session; a second concurrent send on the
    #: same session is rejected by the API layer rather than queued here.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    #: Number of turns evicted from the verbatim window so far.
    evicted_turns: int = 0
    #: Index into ``turns``: everything before it is already in ``summary``.
    #: Without this the same early turns would be folded into the rolling
    #: summary again on every subsequent eviction and it would drift.
    summarised_upto: int = 0

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, ttl: float | None = None) -> bool:
        ttl = settings.session_ttl_seconds if ttl is None else ttl
        return (time.time() - self.last_active) > ttl

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "intent": self.intent.value,
            "facts": dict(self.facts),
            "summary": self.summary,
            "turns": [t.as_dict() for t in self.turns],
            "turn_count": len(self.turns),
            "evicted_turns": self.evicted_turns,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


class SessionStore:
    """TTL + capacity bounded in-memory session registry.

    Bounded on purpose: an unbounded dict keyed by a client-supplied id is a
    trivial memory-exhaustion vector for a public endpoint. When the store is
    full the least-recently-active session is dropped.
    """

    def __init__(self, *, ttl_seconds: int | None = None, max_sessions: int | None = None) -> None:
        self.ttl_seconds = ttl_seconds or settings.session_ttl_seconds
        self.max_sessions = max_sessions or settings.max_sessions
        self._sessions: dict[str, Session] = {}
        self._guard = asyncio.Lock()

    # -- lifecycle -------------------------------------------------------

    @staticmethod
    def new_id() -> str:
        return "s_" + secrets.token_urlsafe(12)

    async def get_or_create(self, session_id: str | None) -> Session:
        async with self._guard:
            self._evict_expired_locked()

            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            # An unknown id from a reconnecting client is honoured rather than
            # replaced, so a page refresh keeps its id even though the history
            # behind it is gone.
            new_id = session_id if session_id else self.new_id()
            if len(self._sessions) >= self.max_sessions:
                self._evict_lru_locked()
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    async def get(self, session_id: str) -> Session | None:
        async with self._guard:
            session = self._sessions.get(session_id)
            if session and session.is_expired(self.ttl_seconds):
                del self._sessions[session_id]
                return None
            return session

    async def reset(self, session_id: str) -> Session:
        """Drop all state for a session id but keep the id itself."""
        async with self._guard:
            fresh = Session(session_id=session_id)
            self._sessions[session_id] = fresh
            return fresh

    async def delete(self, session_id: str) -> bool:
        async with self._guard:
            return self._sessions.pop(session_id, None) is not None

    # -- introspection ---------------------------------------------------

    async def stats(self) -> dict[str, object]:
        async with self._guard:
            self._evict_expired_locked()
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "total_turns": sum(len(s.turns) for s in self._sessions.values()),
            }

    # -- internals -------------------------------------------------------

    def _evict_expired_locked(self) -> None:
        stale = [sid for sid, s in self._sessions.items() if s.is_expired(self.ttl_seconds)]
        for sid in stale:
            del self._sessions[sid]

    def _evict_lru_locked(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_active)
        del self._sessions[oldest.session_id]
