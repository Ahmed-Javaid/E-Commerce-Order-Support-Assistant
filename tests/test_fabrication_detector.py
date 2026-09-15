"""Regression tests for the output-verification fabrication detector.

The detector in ``app/domain/verification.py`` is the last thing standing
between the customer and invented order data -- the single most important
failure mode in this domain, and the one the prompt alone never fully
eliminated. It gates every reply in an order-touching lane, so it is worth
testing in its own right: an earlier version missed the real transcript
*"your order is now in the IN TRANSIT phase"*, and so reported a pass on the
exact behaviour it exists to catch.

Both directions matter. A detector that fired on legitimate policy talk
("once an order is dispatched it cannot be cancelled") would replace correct
answers with a fallback, which is worse than having no detector at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.verification import find_fabrication  # noqa: E402


def fabricates(text: str) -> bool:
    return find_fabrication(text) is not None


@pytest.mark.parametrize(
    "reply",
    [
        # Captured verbatim from a real transcript; the original detector missed it.
        "Since your order is now in the IN TRANSIT phase, any issues would need a return.",
        "Your parcel has been dispatched on Tuesday, October 5th, from our Lahore warehouse.",
        "Currently, your order is out for delivery.",
        "Your order is currently in transit.",
        "Your package has been delivered.",
        "The tracking shows it left the warehouse yesterday.",
        "I can see your order was packed yesterday.",
        "I have checked your account and the refund is processing.",
        "Our system shows the parcel is delayed.",
        "It will arrive by Friday.",
    ],
)
def test_detects_invented_order_data(reply: str) -> None:
    assert fabricates(reply), "fabricated order state slipped past the detector"


@pytest.mark.parametrize(
    "reply",
    [
        "A parcel is only declared lost after 10 business days past the last scan.",
        "Once an order is dispatched it cannot be cancelled; you can refuse it at the door.",
        "Standard shipping takes 4-6 business days from dispatch.",
        "If your tracking page still says label created, that is normal for up to 48 hours.",
        "Could you tell me what your tracking page currently shows?",
        "Orders move from PLACED to PACKED to DISPATCHED before they are in transit.",
        "Express shipping is 2-3 business days and costs PKR 600.",
        "Returns are accepted within 30 calendar days of the delivery date.",
        "I cannot see your order, so I cannot tell you where the parcel is right now.",
        "Delivered means the carrier has marked the parcel as handed over.",
    ],
)
def test_does_not_fire_on_legitimate_policy_talk(reply: str) -> None:
    assert not fabricates(reply), "detector fired on a correct, policy-grounded answer"


@pytest.mark.parametrize(
    "reply",
    [
        # Every one of these was intercepted by an earlier version of the
        # detector during a real evaluation run, replacing a correct answer with
        # the fallback. They are the regression cases that matter most: a
        # false positive silently degrades working behaviour, where a false
        # negative merely fails to improve it.
        "Could you confirm the email address your order was placed with?",
        "Cancelling your order is free while it is still in the PLACED or PACKED state.",
        "Your order was placed using the email on the account, is that right?",
        "Once your order was dispatched, it can no longer be cancelled.",
    ],
)
def test_does_not_fire_on_previously_misfired_phrasings(reply: str) -> None:
    assert not fabricates(reply), "known false positive has regressed"


def test_conditional_framing_is_not_an_assertion() -> None:
    """The discriminator: a policy rule vs a claim about this customer's order."""
    assert not fabricates("Once an order is dispatched it cannot be cancelled.")
    assert fabricates("Your order is dispatched and on its way to you.")


# -- verified orders --------------------------------------------------------
#
# Once the order book existed, "is this fabrication?" stopped being a property
# of the sentence alone and became a property of (sentence, identity). These
# pin the whole matrix.

SARA = "sara.k@example.com"   # NIM-40011234, IN TRANSIT, note mentions dispatch
AMIR = "amir@example.com"     # NIM-77881122, DELIVERED


@pytest.mark.parametrize(
    ("reply", "order_id", "email"),
    [
        # The record is in the model's prompt, so confirming it is simply true.
        ("I can confirm that your order is currently in transit.", "NIM-40011234", SARA),
        # A future expectation is not a claim about the current state. Greedy
        # matching used to capture "delivered" here and flag an accurate reply.
        ("Your order is in transit and should be delivered in a few days.", "NIM-40011234", SARA),
        ("Your order was dispatched 10 days ago.", "NIM-40011234", SARA),
        ("Your order was delivered 8 days ago, well inside the window.", "NIM-77881122", AMIR),
    ],
)
def test_verified_order_may_state_its_own_record(reply, order_id, email) -> None:
    assert find_fabrication(reply, order_id, email) is None


@pytest.mark.parametrize(
    ("reply", "order_id", "email", "why"),
    [
        ("Your order has been delivered already.", "NIM-40011234", SARA, "contradicts IN TRANSIT"),
        ("Your order is out for delivery.", "NIM-77881122", AMIR, "contradicts DELIVERED"),
        ("Your order was dispatched on Tuesday.", "NIM-40011234", SARA, "no record carries a named day"),
        ("Your order is currently in transit.", "NIM-99999999", "ghost@example.com", "order does not exist"),
        ("Your order is currently in transit.", "NIM-40011234", "attacker@example.com", "email does not match"),
        ("Your order is currently in transit.", None, None, "no identity at all"),
        ("I can confirm that your order shipped.", "NIM-99999999", "ghost@example.com", "cannot confirm an unknown order"),
    ],
)
def test_unsupported_claims_are_still_blocked(reply, order_id, email, why) -> None:
    assert find_fabrication(reply, order_id, email) is not None, why


def test_a_real_reference_alone_is_not_enough() -> None:
    """Anyone could guess a reference; the email is what proves ownership."""
    claim = "Your order is currently in transit."
    assert find_fabrication(claim, "NIM-40011234", SARA) is None
    assert find_fabrication(claim, "NIM-40011234", None) is not None
    assert find_fabrication(claim, "NIM-40011234", "someone.else@example.com") is not None


# -- return-window arithmetic ----------------------------------------------
#
# A different failure class from state fabrication, and one the state checks are
# blind to. Asked about an order delivered 41 days ago the model answered "you
# are within the 30-day return window since it was delivered 41 days ago" --
# self-contradictory inside one sentence. Each record states eligibility
# explicitly so the contradiction is checkable without parsing English.

ZOYA = ("NIM-55220147", "zoya@example.com")   # returnable: no  (41 days)
AMIR_O = ("NIM-77881122", "amir@example.com") # returnable: yes (8 days)


@pytest.mark.parametrize(
    ("reply", "order", "why"),
    [
        ("You are within the 30-day return window since it was delivered 41 days ago.",
         ZOYA, "the real observed failure"),
        ("You can still return it under the 30-day window.", ZOYA, "claims eligibility"),
        ("This order is outside the 30-day return window.", AMIR_O, "denies a valid return"),
        ("You cannot return it.", AMIR_O, "denies a valid return"),
    ],
)
def test_return_eligibility_contradictions_are_caught(reply, order, why) -> None:
    assert find_fabrication(reply, *order) is not None, why


@pytest.mark.parametrize(
    ("reply", "order"),
    [
        ("This order is past the 30-day return window, so a return is not possible.", ZOYA),
        ("A fault would still be covered by the 12-month warranty.", ZOYA),
        ("You are within the 30-day return window, delivered 8 days ago.", AMIR_O),
        # Generic policy statement, no order attached.
        ("The return window is 30 calendar days from delivery.", (None, None)),
    ],
)
def test_correct_return_verdicts_pass(reply, order) -> None:
    assert find_fabrication(reply, *order) is None


@pytest.mark.parametrize(
    "reply",
    [
        # Denying an unknown order is the *correct* answer. An earlier version
        # blocked these for containing "I can confirm", replacing a good reply
        # with the fallback.
        "I can confirm that there isn't an order with that reference.",
        "I cannot find an order with that reference.",
        "We have no record of that order.",
        "That order does not exist in our system.",
    ],
)
def test_denials_are_not_fabrication(reply: str) -> None:
    assert find_fabrication(reply, "NIM-99999999", "ghost@example.com") is None


def test_positive_claims_about_unknown_orders_still_blocked() -> None:
    assert find_fabrication(
        "I can confirm that your order is on its way.", "NIM-99999999", "ghost@example.com"
    ) is not None
