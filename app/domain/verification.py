"""Output verification: catching invented order data before the customer sees it.

Fabricating order state -- "your parcel is out for delivery", "dispatched on
Tuesday the 5th" -- is the single worst failure this assistant can produce. It
is confident, specific, plausible, and completely made up, and a customer has no
way to tell it apart from a real answer.

Three rounds of prompt engineering reduced it a lot but never to zero, which is
the honest situation with a 3B model: the system prompt is a *strong preference*,
not a guarantee. So the last line of defence is deterministic. Every generated
reply in an order-touching lane is checked against the patterns below, and a
reply that claims live visibility is replaced with a safe, policy-grounded one
rather than shown.

The patterns are deliberately narrow. A detector that fired on legitimate policy
talk ("once an order is dispatched it cannot be cancelled") would be worse than
none, so both directions are regression-tested in
``tests/test_fabrication_detector.py``.
"""

from __future__ import annotations

import re

from app.domain.knowledge import BRAND

#: Order-state vocabulary. Matching a bare state name is not enough -- the
#: patterns below only fire when a state is *attributed to the customer's order*.
#:
#: PLACED and PACKED are deliberately absent. They collide with ordinary English
#: ("the email the order was placed with", "your order was placed on Tuesday" is
#: the customer's own statement), and asserting them is not the harmful failure
#: anyway: nobody is misled by being told their order exists. The states kept
#: here are the ones that imply *live tracking visibility* the assistant does
#: not have.
_ORDER_STATES = (
    r"(dispatched|in[ -]?transit|out for delivery|delivered|on hold|"
    r"return in progress|refunded|cancelled)"
)

#: Words that make a clause conditional, definitional or policy-general rather
#: than an assertion about this customer's order right now. "Once an order is
#: dispatched it cannot be cancelled" is correct policy; "your order is
#: dispatched" is a fabrication. Without this distinction the detector replaced
#: correct answers with the fallback -- strictly worse than having no detector.
_NON_ASSERTION_MARKERS = re.compile(
    r"\b(once|if|when|whenever|while|unless|until|whether|in case|should it|"
    r"cannot be|can be|can only|is only|are only|would be|will be able|"
    r"policy|states?\b|status(es)?\b|means?\b|typically|usually|normally)\b",
    re.IGNORECASE,
)

#: Sentence splitter used to scope the conditional check to the clause that
#: actually matched, rather than the whole reply.
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")

FABRICATION_SOURCES: tuple[str, ...] = (
    # "...dispatched on Tuesday", "will arrive by the 5th"
    r"\b(arriv\w+|deliver\w+|dispatch\w+|shipp?ed)\s+(on|by)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?\b|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)",
    # "your order is / was / has been / is now in the IN TRANSIT phase"
    r"\byour (parcel|order|package|item|shipment)\b[^.?!]{0,40}\b(is|was|has been)\b"
    r"[^.?!]{0,30}" + _ORDER_STATES,
    # "since your order is currently in transit ..."
    r"\b(since|as|because)\b[^.?!]{0,30}\border\b[^.?!]{0,25}\b(is|was)\b[^.?!]{0,25}"
    + _ORDER_STATES,
    # claiming to have looked something up
    r"\btracking (shows|says|indicates|confirms)\b",
    r"\bI (can see|see|have checked|checked|looked up|pulled up|can confirm)\b"
    r".{0,25}\b(your )?(order|parcel|account|system)",
    r"\b(our|the) (system|records) (shows?|indicates?|says)\b",
)

FABRICATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in FABRICATION_SOURCES
)

#: Shown instead of a reply that claimed live order visibility.
#:
#: Written to still be useful rather than to just refuse: it says what is
#: actually knowable from policy, and ends with a question that keeps the
#: conversation moving. A bare "I cannot help with that" would turn every
#: correction into a dead end.
_FALLBACK_BODY = (
    f"I should be straight with you: I cannot see your order from here, so I cannot "
    f"tell you where the parcel is or when it will arrive -- only your tracking page "
    f"shows that. What I can give you is the {BRAND} policy. A parcel is normally "
    f"scanned within 48 hours of dispatch, standard delivery is 4-6 business days "
    f"after that, and we only treat a parcel as lost once 10 business days have "
    f"passed with no new scan."
)

#: Appended when the conversation still needs identification, so a correction
#: does not derail the stage machine by dropping the question the stage was
#: supposed to ask.
_FALLBACK_IDENTIFY = (
    " To take this further I will need your order reference (it looks like "
    "NIM-12345678) and the email address the order was placed with."
)

_FALLBACK_PROMPT = " What does your tracking page show at the moment?"


def safe_fallback(needs_identification: bool = False) -> str:
    """The replacement text for a reply that claimed live order visibility."""
    if needs_identification:
        return _FALLBACK_BODY + _FALLBACK_IDENTIFY
    return _FALLBACK_BODY + _FALLBACK_PROMPT


#: Convenience constant for the common case (identification already complete).
SAFE_FALLBACK = safe_fallback()


def find_fabrication(text: str) -> str | None:
    """Return the offending span if the reply claims live order visibility.

    Works sentence by sentence. A sentence that matches a fabrication pattern is
    only reported if it also reads as an *assertion* -- a sentence carrying a
    conditional or policy-general marker ("once an order is dispatched...",
    "delivered means...") is correct policy talk and is left alone.

    Returns ``None`` for a clean reply, so the caller can treat it as a simple
    truthiness check while still being able to log exactly what matched.
    """
    for sentence in _SENTENCE_SPLIT.split(text):
        for pattern in FABRICATION_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            if _NON_ASSERTION_MARKERS.search(sentence):
                # Conditional or definitional: correct policy talk, not a claim
                # about this customer's order.
                continue
            return match.group(0)
    return None
