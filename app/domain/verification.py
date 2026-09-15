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

from app.domain.knowledge import BRAND, ORDERS

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
#: "state" and "status" used to be in this list, to spare definitional text. They
#: were too common and became a bypass: the model wrote "this order is currently
#: in the DELIVERED side state" -- a flat contradiction of the record -- and the
#: word "state" alone excused the whole sentence. Definitional text is already
#: covered by "means", "policy" and the conditional words.
_NON_ASSERTION_MARKERS = re.compile(
    r"\b(once|if|when|whenever|while|unless|until|whether|in case|should it|"
    r"cannot be|can be|can only|is only|are only|would be|will be able|"
    r"policy|means?|typically|usually|normally)\b",
    re.IGNORECASE,
)

#: Status names the contradiction check compares against. Kept separate from the
#: regex vocabulary because this list is matched against a record, not a pattern.
_KNOWN_STATES: tuple[str, ...] = (
    "placed", "packed", "dispatched", "in transit", "out for delivery",
    "delivered", "on hold", "return in progress", "refunded", "cancelled",
)

#: Claims that the order CAN be returned. Checked only against a record whose
#: `returnable` field starts "no".
_RETURN_ELIGIBLE_RE = re.compile(
    r"\b(within|inside) the [^.?!]{0,18}(return )?window"
    r"|\byou (are|is) (still )?(eligible|able) to return"
    r"|\b(is|are) (still )?(eligible|elligible) for a (return|refund)"
    r"|\byou can (still )?return (it|this|your)",
    re.IGNORECASE,
)

#: Claims that the order CANNOT be returned. Checked only against a record whose
#: `returnable` field starts "yes".
_RETURN_INELIGIBLE_RE = re.compile(
    r"\b(outside|past|beyond) the [^.?!]{0,18}(return )?window"
    r"|\b(not|no longer) eligible for a (return|refund)"
    r"|\byou cannot return (it|this|your)",
    re.IGNORECASE,
)

#: Phrases that turn a lookup claim into a *denial*. Saying "I cannot find an
#: order with that reference" is the correct answer for an unknown order, and an
#: earlier version blocked it for containing "I can confirm".
_DENIAL_MARKERS = re.compile(
    r"\b(isn.t|is not|aren.t|are not|no order|not an order|cannot find|can.t find|"
    r"could not find|couldn.t find|do not have|don.t have|no record|not in our|"
    r"unable to find|does not exist|doesn.t exist|no matching)\b",
    re.IGNORECASE,
)

#: Sentence splitter used to scope the conditional check to the clause that
#: actually matched, rather than the whole reply.
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")

# The patterns are split by *what they claim*, because a verified order changes
# the verdict for some of them and not others.

#: Claims of lookup ability. Fabrication when we cannot identify the order --
#: but once the reference and email match a record that is sitting in the
#: model's prompt, "I can confirm your order..." is simply true, and blocking it
#: replaced correct answers with a refusal.
_LOOKUP_CLAIM_SOURCES: tuple[str, ...] = (
    r"\btracking (shows|says|indicates|confirms)\b",
    r"\bI (can see|see|have checked|checked|looked up|pulled up|can confirm)\b"
    r".{0,25}\b(your )?(order|parcel|account|system)",
    r"\b(our|the) (system|records) (shows?|indicates?|says)\b",
)

#: Assertions of a specific state, with the state captured so it can be compared
#: against the record rather than merely detected.
#
# The gaps are LAZY ({0,30}? rather than {0,30}). Greedy matching captured the
# furthest state in the sentence, so "your order is in transit and should be
# delivered in a few days" captured "delivered" -- a future expectation -- and
# flagged an accurate reply as contradicting the record. Lazy captures the state
# actually being asserted.
#: How the assistant refers to the order. Anchoring on "your" alone was not
#: enough: against the real order book the model wrote "this order is currently
#: in the DELIVERED side state" and "since this order has been dispatched", both
#: flatly contradicting the record, and both sailed past the narrower pattern.
_ORDER_SUBJECT = (
    r"(?:your|this|that|the)\s+(?:parcel|order|package|item|shipment)"
    r"|order\s+NIM-\d{8}"
)

#: Verb phrases that assert a state, including the perfect and progressive forms
#: the model actually reaches for ("has been", "is now in").
_ASSERT_VERB = r"(?:is|was|are|were|has been|have been|is now in|was in)"

_STATE_ASSERTION_SOURCES: tuple[str, ...] = (
    rf"\b(?:{_ORDER_SUBJECT})\b[^.?!]{{0,45}}?\b{_ASSERT_VERB}\b[^.?!]{{0,35}}?"
    + _ORDER_STATES,
    rf"\b(?:since|as|because)\b[^.?!]{{0,30}}?\b(?:{_ORDER_SUBJECT})\b"
    rf"[^.?!]{{0,25}}?\b{_ASSERT_VERB}\b[^.?!]{{0,25}}?" + _ORDER_STATES,
)

#: A named day or date. Never supported: no record carries one, so this is
#: invented whether or not the order is verified.
_DATE_CLAIM_SOURCES: tuple[str, ...] = (
    r"\b(arriv\w+|deliver\w+|dispatch\w+|shipp?ed)\s+(on|by)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?\b|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)",
)

FABRICATION_SOURCES: tuple[str, ...] = (
    _DATE_CLAIM_SOURCES + _STATE_ASSERTION_SOURCES + _LOOKUP_CLAIM_SOURCES
)

_LOOKUP_CLAIM_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in _LOOKUP_CLAIM_SOURCES)
_STATE_ASSERTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in _STATE_ASSERTION_SOURCES)
_DATE_CLAIM_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in _DATE_CLAIM_SOURCES)

FABRICATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    _DATE_CLAIM_PATTERNS + _STATE_ASSERTION_PATTERNS + _LOOKUP_CLAIM_PATTERNS
)

#: Shown instead of a reply that claimed live order visibility.
#:
#: Written to still be useful rather than to just refuse: it says what is
#: actually knowable from policy, and ends with a question that keeps the
#: conversation moving. A bare "I cannot help with that" would turn every
#: correction into a dead end.
_FALLBACK_BODY = (
    f"I should be straight with you: I cannot confirm anything about that order. "
    f"I can only look at an order once I have the reference and the email it was "
    f"placed with, and they have to match. What I can give you either way is the "
    f"{BRAND} policy: a parcel is normally scanned within 48 hours of dispatch, "
    f"standard delivery is 4-6 business days after that, and we only treat a parcel "
    f"as lost once 10 business days have passed with no new scan."
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


def _squash(text: str) -> str:
    return text.lower().replace(" ", "").replace("-", "")


def _states_in(text: str) -> set[str]:
    """Order-state names asserted anywhere in a piece of text."""
    squashed = _squash(text)
    return {state for state in _KNOWN_STATES if _squash(state) in squashed}


def _supported_states(record: dict[str, str]) -> set[str]:
    """States a verified order's own record vouches for.

    Both the status field and the free-text note count: an IN TRANSIT order whose
    note reads "Dispatched 10 days ago" legitimately supports a reply that
    mentions dispatch.
    """
    return _states_in(record["status"] + " " + record.get("note", ""))


def is_verified(order_id: str | None, email: str | None) -> bool:
    """True when the customer has proved which order they mean.

    Both halves are required, exactly as the policy states: a reference that
    exists in the book is not enough on its own, because anyone could guess a
    reference. The email must match the one the order was placed with.
    """
    if not order_id or not email:
        return False
    record = ORDERS.get(order_id.upper())
    return bool(record) and record["email"].lower() == email.strip().lower()


def find_fabrication(
    text: str, order_id: str | None = None, email: str | None = None
) -> str | None:
    """Return the offending span if the reply claims order knowledge it lacks.

    Three cases, and the distinction is the whole point:

    * **Verified order** (reference in the book *and* matching email) -- the
      record is sitting in the model's prompt, so stating its status is correct
      behaviour, not fabrication. We only object if the reply names a status
      that *contradicts* the record.
    * **Unverified** -- no reference, a wrong email, or a reference that is not
      in the book. Any assertion about where the parcel is or what was
      dispatched is invented, and is blocked.
    * **Policy talk in either case** -- a sentence carrying a conditional or
      definitional marker ("once an order is dispatched...", "delivered means
      ...") is correct and is always left alone.

    Returns ``None`` for a clean reply, so callers can treat it as a truthiness
    check while still being able to log exactly what matched.
    """
    verified = is_verified(order_id, email)
    record = ORDERS.get((order_id or "").upper()) if verified else None

    for sentence in _SENTENCE_SPLIT.split(text):
        if _NON_ASSERTION_MARKERS.search(sentence):
            # Conditional or definitional: correct policy talk in every case.
            continue

        # A named day or date is never in any record, so it is invented whether
        # or not we know which order this is.
        for pattern in _DATE_CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if match:
                return match.group(0)

        if record is None:
            # Unidentified: any claim of lookup ability, or of a state, is made up
            # -- unless it is a *denial*. "I can confirm that there isn't an order
            # with that reference" is exactly the behaviour we want for an unknown
            # order, and blocking it replaced a good answer with the fallback.
            if _DENIAL_MARKERS.search(sentence):
                continue
            for pattern in _LOOKUP_CLAIM_PATTERNS + _STATE_ASSERTION_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    return match.group(0)
            continue

        # Verified: the record is in the model's prompt, so confirming and
        # describing it is correct. Object only when the asserted state is one
        # the record does not support -- which would mean the model ignored the
        # record it was handed. Compared against the status *and* the note,
        # because an IN TRANSIT order whose note says "dispatched 10 days ago"
        # legitimately supports a reply that mentions dispatch.
        supported = _supported_states(record)
        for pattern in _STATE_ASSERTION_PATTERNS:
            match = pattern.search(sentence)
            if match and _squash(match.group(1)) not in {_squash(s) for s in supported}:
                return match.group(0)

        # Return eligibility is arithmetic, and arithmetic is where a 3B model
        # slips: asked about an order delivered 41 days ago it replied "you are
        # within the 30-day return window since it was delivered 41 days ago" --
        # self-contradictory in a single sentence, and invisible to a check that
        # only looks at order states. The record states eligibility explicitly so
        # the contradiction can be caught without parsing English.
        verdict = record.get("returnable", "").split()[0].lower()
        if verdict == "no":
            match = _RETURN_ELIGIBLE_RE.search(sentence)
            if match:
                return match.group(0)
        elif verdict == "yes":
            match = _RETURN_INELIGIBLE_RE.search(sentence)
            if match:
                return match.group(0)
    return None
