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
