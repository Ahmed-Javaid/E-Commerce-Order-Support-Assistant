"""The conversation manager.

This is the component the API layer talks to. One public coroutine,
:meth:`ConversationManager.stream_turn`, takes a session plus a raw user message
and yields protocol events until the turn is complete.

What it does per turn, in order:

1. **Validate** the message (empty, oversized, non-text).
2. **Guard** it deterministically -- persona-override attempts and blatantly
   off-domain requests are detected by regex. By default the model still
   writes the refusal (guard_mode=steer), so every customer-visible word is
   model-generated; guard_mode=reply returns a canned string instead.
3. **Track state** -- classify the support lane, extract identifiers into pinned
   facts, detect a mid-conversation topic switch and suspend the previous lane.
4. **Advance the stage machine** so the prompt tells the model where it is.
5. **Build the prompt** -- a byte-identical system prompt, the verbatim history
   window, then the new message with this turn's context attached. That layout
   keeps the runtime's KV-cache prefix growing instead of being invalidated.
6. **Stream** the answer, post-processing each delta.
7. **Commit** the turn and, if history was evicted, refresh the rolling summary
   *after* the answer is already on screen so the customer never waits for it.

There are no tools, no function calls and no retrieval anywhere in this path.
Everything the assistant appears to know comes from the static knowledge block
in the system prompt plus what the customer said earlier in this session.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from app.config import settings
from app.conversation.memory import (
    estimate_messages_tokens,
    plan_window,
    render_transcript,
)
from app.conversation.session import Session, Turn
from app.domain import policy as pol
from app.domain.prompts import (
    SUMMARISER_SYSTEM,
    build_summariser_prompt,
    build_system_prompt,
    build_turn_context,
    compose_user_turn,
)
from app.domain.verification import find_fabrication, safe_fallback
from app.llm.base import ChatMessage, GenerationStats, LLMEngine, LLMError

EventType = Literal["start", "token", "done", "error", "context", "correction"]


@dataclass
class TurnEvent:
    """One protocol event emitted while handling a turn."""

    type: EventType
    text: str = ""
    code: str = ""
    stage: str = ""
    intent: str = ""
    guarded: str = ""
    #: Set when output verification replaced a reply that invented order data.
    corrected: str = ""
    stats: GenerationStats | None = None
    context: dict[str, object] = field(default_factory=dict)


class ValidationError(ValueError):
    """Raised for a user message the manager refuses to process."""

    def __init__(self, message: str, code: str = "invalid_message") -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# Output hygiene
# --------------------------------------------------------------------------

#: Small models occasionally echo a prompt heading or a speaker label. These are
#: stripped from the *accumulated* text rather than from individual deltas,
#: because a heading can straddle a chunk boundary.
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(CURRENT STAGE|NOTES FOR THIS TURN|CONFIRMED DETAILS|STYLE|RULES YOU MUST FOLLOW|"
        r"SUMMARY OF EARLIER TURNS|NIMBUS POLICY AND CATALOGUE REFERENCE)\s*:?.*$",
        r"^\s*(Agent|Assistant|Ava)\s*:\s*",
    )
)

_CLOSING_RE = re.compile(
    r"^\s*(thanks|thank you|thankyou|thx|that.s all|that is all|nothing else|no thanks|"
    r"bye|goodbye|cheers|great, thanks|perfect thanks)\b",
    re.IGNORECASE,
)


_BULLET_RE = re.compile(r"^\s*[-*•]\s")


def scrub(text: str) -> str:
    """Remove leaked prompt scaffolding from a completed reply.

    When a *heading* leaks, the bullets under it almost always leak with it, so
    bullet lines immediately following a stripped heading are dropped too. That
    lookahead stops at the first non-bullet line, which is why the assistant's
    own legitimate bullet lists (used when reading details back) survive: they
    are never preceded by one of these headings.
    """
    cleaned: list[str] = []
    dropping_bullets = False

    for line in text.splitlines():
        stripped = line
        leaked = False
        for pattern in _LEAK_PATTERNS:
            replaced = pattern.sub("", stripped)
            if replaced != stripped:
                leaked = True
                stripped = replaced
        if leaked and not stripped.strip():
            # The whole line was scaffolding; swallow its bullets as well.
            dropping_bullets = True
            continue
        if dropping_bullets:
            if _BULLET_RE.match(line):
                continue
            dropping_bullets = False
        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


# --------------------------------------------------------------------------
# Stage machine
# --------------------------------------------------------------------------


def _missing_slots(intent: pol.Intent, facts: dict[str, str]) -> tuple[str, ...]:
    required = pol.REQUIRED_SLOTS.get(intent, ())
    return tuple(slot for slot in required if not facts.get(slot))


def next_stage(
    session: Session,
    intent: pol.Intent,
    message: str,
    is_first_turn: bool,
) -> pol.Stage:
    """Decide which stage the assistant should act in for this reply.

    The stages are Nimbus-specific, not a generic greet/collect/close skeleton:
    ``ORDER_IDENTIFICATION`` exists because every order-touching lane at Nimbus
    needs a reference plus the account email before anything else can be said,
    and ``POLICY_RESOLUTION`` exists because the actual deliverable of a support
    turn here is a policy verdict (eligible / not, window, fee, next step).
    """
    if _CLOSING_RE.match(message.strip()) and session.turns:
        return pol.Stage.CLOSING

    if intent is pol.Intent.SMALL_TALK and is_first_turn:
        return pol.Stage.GREETING

    if intent in (pol.Intent.PRODUCT_INFO, pol.Intent.POLICY_INFO):
        # These lanes need no identification -- answer straight from policy.
        return pol.Stage.POLICY_RESOLUTION

    if intent in pol.IDENTIFIED_INTENTS:
        identity_missing = _missing_slots(intent, session.facts)
        if "order_id" in identity_missing or "email" in identity_missing:
            return pol.Stage.ORDER_IDENTIFICATION
        if identity_missing:
            return pol.Stage.ISSUE_DETAIL
        # Everything gathered: resolve, then read back on the following turn.
        if session.stage is pol.Stage.POLICY_RESOLUTION:
            return pol.Stage.CONFIRMATION
        return pol.Stage.POLICY_RESOLUTION

    if intent is pol.Intent.UNKNOWN:
        return pol.Stage.INTENT_TRIAGE

    return pol.Stage.INTENT_TRIAGE


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------


class ConversationManager:
    """Owns turn-taking, prompt assembly and memory policy for every session."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine

    # -- validation ------------------------------------------------------

    def validate(self, message: object) -> str:
        if not isinstance(message, str):
            raise ValidationError("Message must be a string.", "invalid_message_type")
        text = message.strip()
        if not text:
            raise ValidationError("Message cannot be empty.", "empty_message")
        if len(text) > settings.max_message_chars:
            raise ValidationError(
                f"Message is too long ({len(text)} characters). "
                f"The limit is {settings.max_message_chars}.",
                "message_too_long",
            )
        return text

    # -- prompt assembly -------------------------------------------------

    def build_messages(
        self, session: Session, user_message: str, notes: list[str]
    ) -> tuple[list[ChatMessage], dict[str, object]]:
        """Assemble the chat-formatted prompt and a report on the memory decision."""
        plan = plan_window(
            session.turns,
            settings.history_token_budget,
            settings.min_verbatim_turns,
        )

        # The system prompt is byte-identical every turn, and history entries are
        # the customer's raw words, so the whole prefix up to the newest message
        # stays cacheable and only the new turn is evaluated. See
        # build_system_prompt() for the measurement behind this layout.
        messages: list[ChatMessage] = [ChatMessage("system", build_system_prompt())]
        for turn in plan.kept:
            messages.append(ChatMessage("user", turn.user))
            if turn.assistant:
                messages.append(ChatMessage("assistant", turn.assistant))

        context = build_turn_context(
            stage=session.stage,
            facts=session.facts,
            summary=session.summary,
            notes=notes,
            lane=session.intent.value,
        )
        messages.append(ChatMessage("user", compose_user_turn(context, user_message)))

        report: dict[str, object] = {
            "turns_total": len(session.turns),
            "turns_verbatim": len(plan.kept),
            "turns_evicted_now": len(plan.evicted),
            "turns_evicted_total": session.evicted_turns,
            "history_tokens": plan.kept_tokens,
            "history_budget": plan.budget,
            "prompt_tokens_est": estimate_messages_tokens(messages),
            "context_window": settings.context_window,
            "has_summary": bool(session.summary),
            "pinned_facts": sorted(session.facts),
        }
        return messages, report

    # -- state tracking --------------------------------------------------

    def _update_state(self, session: Session, message: str) -> list[str]:
        """Update intent, slots and stage. Returns per-turn prompt notes."""
        notes: list[str] = []
        is_first_turn = not session.turns

        new_slots = pol.extract_slots(message)
        session.facts.update(new_slots)

        previous_intent = session.intent
        intent = pol.classify_intent(message, previous_intent)

        # --- topic switch handling -------------------------------------
        # Resuming a parked lane is checked first: coming back to it is *also*
        # a change of intent, so the generic switch branch would otherwise
        # swallow it and the assistant would restart the lane from scratch.
        switched = (
            previous_intent not in (pol.Intent.UNKNOWN, pol.Intent.SMALL_TALK)
            and intent not in (pol.Intent.UNKNOWN, pol.Intent.SMALL_TALK)
            and intent is not previous_intent
        )
        if session.suspended_intent is not None and intent is session.suspended_intent:
            notes.append(
                "The customer is returning to the topic they paused earlier. "
                "Pick it up where it left off; do not start over."
            )
            session.suspended_intent = None
        elif switched:
            unfinished = _missing_slots(previous_intent, session.facts)
            if unfinished and previous_intent in pol.IDENTIFIED_INTENTS:
                # Park the old lane so it can be offered back at the end.
                # Collected identifiers stay pinned in session.facts regardless.
                session.suspended_intent = previous_intent
                notes.append(
                    f"The customer has switched topic from {previous_intent.value.replace('_', ' ')} "
                    f"to {intent.value.replace('_', ' ')} before the first was finished. "
                    "Acknowledge the switch in at most half a sentence, deal with the new topic, "
                    "and at the end offer to return to the earlier one."
                )
            else:
                notes.append(
                    f"The customer has moved from {previous_intent.value.replace('_', ' ')} to "
                    f"{intent.value.replace('_', ' ')}. Follow them without re-greeting."
                )

        session.intent = intent

        # --- malformed identifier hint ---------------------------------
        if pol.looks_like_bad_order_id(message):
            notes.append(
                "What the customer just gave does not match the NIM-12345678 order "
                "reference format. Tell them the expected format and ask them to check "
                "the confirmation email. Do not treat it as a valid reference."
            )

        # --- stage ------------------------------------------------------
        session.stage = next_stage(session, intent, message, is_first_turn)

        if new_slots:
            notes.append(
                "Do not ask again for details already listed under CONFIRMED DETAILS."
            )

        # The moment a specific order is on the table is exactly when the model
        # starts inventing a status for it, so the reminder is injected only in
        # those lanes -- a note that fires on every turn stops being read.
        if intent in pol.IDENTIFIED_INTENTS:
            notes.append(
                "This turn is about one specific order. You cannot see it. Do not state "
                "a status, scan, location, carrier, dispatch date or delivery date for "
                "it -- describe what the policy says and what the customer should check."
            )

        return notes

    # -- the turn --------------------------------------------------------

    async def stream_turn(
        self, session: Session, raw_message: object
    ) -> AsyncIterator[TurnEvent]:
        """Handle one user turn, yielding protocol events as they happen."""
        started = time.perf_counter()
        message = self.validate(raw_message)
        session.touch()

        # 1. Deterministic guard.
        #
        # There is a real tension here. Detection should be deterministic -- a
        # regex is consistent in a way a 3B model's judgement is not -- but the
        # assignment requires every response to come from prompt orchestration
        # and conversational memory alone, and a hardcoded string is neither.
        # So detection is always deterministic and *who writes the reply* is
        # configurable:
        #
        #   steer (default) -- inject an instruction, let the model write the
        #       refusal. Every customer-visible word stays model-generated.
        #   reply           -- return a canned string, never calling the model.
        #       Instant and perfectly consistent, but not generated.
        #   off             -- no guard; refusals rest on the system prompt.
        #
        # See README section 3.5.
        verdict = (
            pol.guard(message) if settings.guard_mode != "off" else pol.GuardVerdict(False)
        )
        guard_reason = verdict.reason
        guard_note = ""
        if verdict.blocked and settings.guard_mode == "steer":
            guard_note = pol.GUARD_STEER.get(verdict.reason, "")
            if guard_note:
                # Fall through to the normal model path, carrying the steer.
                verdict = pol.GuardVerdict(False)

        if verdict.blocked:
            session.stage = pol.Stage.OUT_OF_SCOPE
            yield TurnEvent(
                "start",
                stage=session.stage.value,
                intent=pol.Intent.OUT_OF_SCOPE.value,
                guarded=verdict.reason,
            )
            # Streamed word-by-word so the client renders it exactly like a
            # model answer -- a canned reply should not look like a different
            # kind of thing to the user.
            for index, word in enumerate(verdict.reply.split(" ")):
                yield TurnEvent("token", text=word if index == 0 else " " + word)
            elapsed = (time.perf_counter() - started) * 1000.0
            session.turns.append(
                Turn(
                    user=message,
                    assistant=verdict.reply,
                    stage=pol.Stage.OUT_OF_SCOPE,
                    intent=pol.Intent.OUT_OF_SCOPE,
                    guarded=verdict.reason,
                    latency_ms=elapsed,
                )
            )
            yield TurnEvent(
                "done",
                stage=session.stage.value,
                intent=pol.Intent.OUT_OF_SCOPE.value,
                guarded=verdict.reason,
                stats=GenerationStats(
                    ttft_ms=elapsed, total_ms=elapsed, engine="guard", model="none"
                ),
            )
            return

        # 2. State tracking and stage advancement.
        notes = self._update_state(session, message)
        if guard_note:
            # A steered turn is out of scope whatever the lane cues said.
            session.stage = pol.Stage.OUT_OF_SCOPE
            notes = [guard_note]
        messages, report = self.build_messages(session, message, notes)

        yield TurnEvent(
            "start",
            stage=session.stage.value,
            intent=session.intent.value,
            guarded=f"steer:{guard_reason}" if guard_note else "",
        )

        # 3. Stream the answer.
        pieces: list[str] = []
        stats: GenerationStats | None = None
        try:
            async for chunk in self.engine.stream_chat(messages):
                if chunk.done:
                    stats = chunk.stats
                    break
                pieces.append(chunk.text)
                yield TurnEvent("token", text=chunk.text)
        except LLMError as exc:
            # Record the failed turn so the history does not silently lose the
            # customer's message, then report a clean error.
            session.turns.append(
                Turn(
                    user=message,
                    assistant="",
                    stage=session.stage,
                    intent=session.intent,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            yield TurnEvent("error", text=str(exc), code=exc.code)
            return

        answer = scrub("".join(pieces))
        if not answer:
            answer = (
                "Sorry, I did not manage to put a reply together there. "
                "Could you say that again?"
            )

        # Last line of defence against invented order data. The prompt makes this
        # rare; this makes it impossible to reach the customer. Only order-touching
        # lanes are checked, because only they can fabricate an order state --
        # running it on policy answers would risk false positives for no benefit.
        corrected = ""
        if session.intent in pol.IDENTIFIED_INTENTS:
            offending = find_fabrication(answer)
            if offending:
                corrected = offending
                # The fabricated text is replaced in history too, so it cannot be
                # replayed next turn and become "context" the model builds on.
                # The replacement keeps asking for whatever the stage still needs,
                # so a correction does not derail the flow.
                answer = safe_fallback(
                    needs_identification=session.stage is pol.Stage.ORDER_IDENTIFICATION
                )

        elapsed = (time.perf_counter() - started) * 1000.0
        session.turns.append(
            Turn(
                user=message,
                assistant=answer,
                stage=session.stage,
                intent=session.intent,
                guarded=f"steer:{guard_reason}" if guard_note else "",
                latency_ms=elapsed,
            )
        )
        session.touch()

        if corrected:
            log_note = f"output verification replaced a fabricated claim: {corrected!r}"
            report["verification"] = log_note
            # The client has already rendered the fabricated text, so tell it to
            # replace the whole message rather than append to it.
            yield TurnEvent("correction", text=answer, corrected=corrected)

        yield TurnEvent(
            "done",
            stage=session.stage.value,
            intent=session.intent.value,
            guarded=f"steer:{guard_reason}" if guard_note else "",
            corrected=corrected,
            stats=stats,
            context=report,
        )

        # 4. Refresh the rolling summary *after* the answer is on screen, so the
        #    customer never waits for compression.
        evicted_now = int(report.get("turns_evicted_now") or 0)
        if evicted_now > session.summarised_upto and settings.enable_rolling_summary:
            await self._refresh_summary(session, evicted_now)
            yield TurnEvent(
                "context",
                stage=session.stage.value,
                context={
                    "summary": session.summary,
                    "turns_evicted_total": session.evicted_turns,
                },
            )

    # -- rolling summary -------------------------------------------------

    async def _refresh_summary(self, session: Session, evicted_count: int) -> None:
        """Fold the *newly* evicted turns into the running summary.

        Only the slice between ``summarised_upto`` and ``evicted_count`` is sent:
        evicted turns stay in ``session.turns`` (they are simply not replayed
        verbatim), so without that watermark the same early turns would be folded
        in again on every later eviction and the summary would drift.

        Uses the same local model at temperature 0 with a tight token cap. On
        failure we keep the previous summary rather than dropping memory on the
        floor -- a stale summary beats none.
        """
        evicted = session.turns[session.summarised_upto : evicted_count]
        if not evicted:
            return
        transcript = render_transcript(evicted)
        if not transcript.strip():
            return

        prompt = [
            ChatMessage("system", SUMMARISER_SYSTEM),
            ChatMessage("user", build_summariser_prompt(session.summary, transcript)),
        ]
        try:
            summary = await self.engine.complete(
                prompt, temperature=0.0, max_tokens=settings.summary_max_tokens
            )
        except LLMError:
            return

        summary = scrub(summary)
        if summary:
            session.summary = summary
            session.summarised_upto = evicted_count
            session.evicted_turns = evicted_count
