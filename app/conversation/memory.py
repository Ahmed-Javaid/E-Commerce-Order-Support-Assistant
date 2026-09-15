"""Context-window management.

The problem this module solves: the prompt grows every turn, but CPU prompt
evaluation is linear in prompt length, so an unbounded history turns a 1.5 s
time-to-first-token into a 15 s one by turn twenty. We need a policy that keeps
the prompt roughly constant-size without the assistant forgetting what the
customer told it.

The scheme is a **three-tier memory**:

1. **Pinned facts** -- identifiers the customer supplied (order reference,
   email, RMA) are extracted into ``Session.facts`` and re-injected into the
   system prompt every turn. They survive eviction entirely and cost ~20 tokens.
   This is what prevents the classic failure where the assistant asks for the
   order number again on turn nine because turn two fell out of the window.
2. **Verbatim recent window** -- the newest turns that fit in
   ``history_token_budget``, always at least ``min_verbatim_turns`` exchanges so
   immediate anaphora ("the second one", "yes, do that") still resolves.
3. **Rolling summary** -- anything evicted from tier 2 is folded into a running
   summary by a short, low-temperature call to the *same* local model. The
   summary is rewritten rather than appended, and capped, so it cannot itself
   grow without bound.

Tiers 1 and 3 are what make this cheaper than a naive sliding window: the
prompt stays flat in size while the facts that actually matter for the next
reply stay available.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.conversation.session import Turn
from app.conversation.tokens import PER_MESSAGE_TOKEN_OVERHEAD, estimate_tokens
from app.llm.base import ChatMessage

__all__ = [
    "WindowPlan",
    "estimate_messages_tokens",
    "estimate_tokens",
    "plan_window",
    "render_transcript",
]


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Approximate the token count of a full chat-formatted prompt."""
    return sum(estimate_tokens(m.content) + PER_MESSAGE_TOKEN_OVERHEAD for m in messages)


@dataclass(frozen=True)
class WindowPlan:
    """The outcome of deciding what to keep.

    ``kept`` and ``evicted`` partition the turn list passed in, preserving order.
    """

    kept: list[Turn]
    evicted: list[Turn]
    kept_tokens: int
    budget: int

    @property
    def evicted_any(self) -> bool:
        return bool(self.evicted)


def plan_window(
    turns: list[Turn],
    budget_tokens: int,
    min_verbatim_turns: int = 2,
) -> WindowPlan:
    """Choose which turns stay verbatim in the prompt.

    Walks backwards from the newest turn, accumulating until the budget is
    exhausted. The newest ``min_verbatim_turns`` exchanges are kept even if that
    overruns the budget -- dropping the turn we are replying to would be worse
    than a slightly long prompt. A single oversized turn is therefore kept
    rather than silently dropped; ``ConversationManager`` caps user input at
    ``max_message_chars`` before it ever reaches here.
    """
    if not turns:
        return WindowPlan([], [], 0, budget_tokens)

    floor = max(0, min_verbatim_turns)
    kept_reversed: list[Turn] = []
    used = 0

    for index, turn in enumerate(reversed(turns)):
        cost = turn.token_cost()
        if index < floor or used + cost <= budget_tokens:
            kept_reversed.append(turn)
            used += cost
        else:
            break

    kept = list(reversed(kept_reversed))
    evicted = turns[: len(turns) - len(kept)]
    return WindowPlan(kept=kept, evicted=evicted, kept_tokens=used, budget=budget_tokens)


def render_transcript(turns: list[Turn]) -> str:
    """Flatten turns into a plain transcript for the summariser."""
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Customer: {turn.user.strip()}")
        if turn.assistant.strip():
            lines.append(f"Agent: {turn.assistant.strip()}")
    return "\n".join(lines)
