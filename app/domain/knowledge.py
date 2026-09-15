"""Static brand knowledge for the Nimbus order-support assistant.

This module is deliberately a *constant*, not a database or an index. It is
concatenated into the system prompt verbatim on every turn, which keeps the
assignment's "no tools, no RAG" constraint satisfied: nothing is retrieved at
query time and nothing is looked up per-user. The whole knowledge surface is
small enough (~600 tokens) to live permanently in the context window.

If this block ever grows past ~800 tokens it should be trimmed rather than
retrieved -- growing it into a searchable store would turn the system into RAG.
"""

from __future__ import annotations

BRAND = "Nimbus"
ASSISTANT_NAME = "Ava"
ASSISTANT_ROLE = "Order Support Specialist"
SUPPORT_HOURS = "Mon-Sat, 09:00-21:00 PKT"
SUPPORT_EMAIL = "support@nimbus.example"

# Order references look like NIM-12345678; RMA references like RMA-4831902.
ORDER_ID_PATTERN = r"\bNIM-\d{8}\b"
RMA_PATTERN = r"\bRMA-\d{7}\b"

POLICY_VERSION = "2026-03"

# --------------------------------------------------------------------------
# The knowledge block itself. Written as terse, declarative bullets: small
# instruction-tuned models follow short imperative lines far more reliably
# than they follow prose paragraphs.
# --------------------------------------------------------------------------

STORE_PROFILE = f"""\
{BRAND} is an online retailer of consumer electronics and smart-home gear.
Catalogue categories: audio (headphones, speakers), wearables (fitness bands,
smartwatches), smart home (plugs, bulbs, cameras, thermostats), laptop and
phone accessories, and power (chargers, power banks).
{BRAND} does NOT sell: clothing, groceries, furniture, medication, vehicles,
software licences, or gift cards.
Support hours: {SUPPORT_HOURS}. Escalation address: {SUPPORT_EMAIL}."""

SHIPPING_POLICY = """\
SHIPPING TIERS (domestic):
- Standard: 4-6 business days. Free over PKR 5,000, otherwise PKR 250.
- Express: 2-3 business days, PKR 600.
- Priority: next business day if ordered before 14:00 PKT, PKR 1,200.
Orders are dispatched from the Lahore warehouse. Cut-off for same-day
dispatch is 14:00 PKT on business days. Remote-area deliveries add 2 business
days.
Business days are Monday to Saturday. There is no dispatch and no delivery on
Sundays or public holidays.
International shipping is not offered, and parcels cannot be collected from a
post office or depot -- every order is delivered to the address on the order.
A tracking link is emailed when the parcel is scanned at the warehouse,
usually within 24 hours of dispatch. "Label created" means the parcel has not
been scanned yet; that is normal for up to 48 hours.
A parcel is only declared lost after 10 business days past the last scan."""

RETURNS_POLICY = """\
RETURNS AND REFUNDS:
- Return window: 30 calendar days from the DELIVERY date.
- Condition: unused, in original packaging, all accessories included.
- Opened audio products (earbuds/headphones) are returnable only if faulty,
  for hygiene reasons.
- Non-returnable: opened software/firmware dongles, clearance items marked
  FINAL SALE, and any item damaged by the customer.
- Faulty items: 12-month manufacturer warranty, handled as a warranty claim
  rather than a return; no 30-day limit.
- Process: support issues an RMA reference (RMA-#######), customer ships the
  item back (free return label for faulty/wrong items, PKR 300 deducted for
  change-of-mind returns).
- Refund timing: 3-5 business days after the warehouse inspects the return.
  Refunds go to the original payment method only. Card refunds can take a
  further 5-7 business days to appear on a statement.
- Exchanges: treated as a return plus a new order; a like-for-like exchange
  keeps the original price even if the item's price has changed."""

ORDER_LIFECYCLE = """\
ORDER STATES (in order): PLACED -> PACKED -> DISPATCHED -> IN TRANSIT ->
OUT FOR DELIVERY -> DELIVERED. Side states: ON HOLD (payment or address
verification), RETURN IN PROGRESS, REFUNDED, CANCELLED.
These are DEFINITIONS of the labels a customer sees on their own tracking page.
You cannot determine which state any order is in. Only the customer can read
that off their tracking page, or tell you. Never assign a state to an order
yourself, and never say an order "is" or "is now in" any of these states.
- Cancellation is free and instant while the order is PLACED or PACKED.
- Once DISPATCHED an order cannot be cancelled; it must be refused at the
  door or returned after delivery.
- Address changes are possible only while PLACED.
- ON HOLD orders need the customer to confirm their address or re-authorise
  payment before they move on."""

PAYMENT_POLICY = """\
PAYMENTS: cards (Visa/Mastercard), bank transfer, and cash on delivery (COD).
COD is capped at PKR 50,000 and is unavailable for remote areas.
A pending card authorisation that did not turn into an order is released by
the bank within 5-7 business days.
Price-match or promo-code disputes are reviewed case by case; a promo code
cannot be applied retroactively to an order already placed."""

IDENTIFIERS = f"""\
IDENTIFIERS:
- Order reference format: NIM-12345678 ({BRAND} + 8 digits). It is printed on
  the order confirmation email.
- Return reference format: RMA-1234567 (RMA + 7 digits).
- To act on an order the customer must supply the order reference AND the
  email address the order was placed with."""

# The single string that gets injected into the system prompt.
KNOWLEDGE_BLOCK = "\n\n".join(
    [
        STORE_PROFILE,
        ORDER_LIFECYCLE,
        SHIPPING_POLICY,
        RETURNS_POLICY,
        PAYMENT_POLICY,
        IDENTIFIERS,
    ]
)
