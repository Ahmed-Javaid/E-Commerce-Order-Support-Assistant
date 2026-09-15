"""Static brand knowledge for the Nimbus order-support assistant.

Everything here is a *constant*, not a database or an index: the brand policy
and a six-order demo book, both concatenated into the system prompt verbatim on
every turn. That keeps the assignment's "no tools, no RAG" constraint satisfied,
because nothing is selected in response to the customer's question -- the model
reads what is already in its context, exactly as it reads the policy text.

The line this must not cross: the moment the order book grows large enough that
we have to *choose* which rows to include, that choice is retrieval and the
system becomes RAG. Six orders, always all present, stays on the safe side.
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
You may state the status of an order ONLY if that exact order reference appears
in the DEMO ORDER BOOK below and the customer has given the matching email. For
any other reference, you do not know the status and must say so instead of
guessing.
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


# --------------------------------------------------------------------------
# Demo order book
# --------------------------------------------------------------------------
#
# Six orders, chosen so that between them they exercise every policy branch:
# in-transit, delivered-and-returnable, delivered-past-window, cancellable,
# already-dispatched, and on-hold.
#
# Why this is not RAG. The entire book is rendered into the system prompt on
# every single turn, exactly like the policy text above. Nothing is selected in
# response to the customer's question, there is no index, and no code fetches a
# record to answer a query -- the model simply reads what is already in its
# context. That is prompt design. It would become retrieval the moment the book
# grew large enough that we had to pick which rows to include, which is the
# reason it is capped at six and lives in a constant rather than a file.

ORDERS: dict[str, dict[str, str]] = {
    "NIM-40011234": {
        "email": "sara.k@example.com",
        "item": "Nimbus Aura 2 wireless earbuds",
        "price": "PKR 8,900",
        "tier": "Express",
        "placed": "12 days ago",
        "status": "IN TRANSIT",
        "returnable": "not yet - not delivered",
        "note": "Dispatched 10 days ago. Last carrier scan was 6 days ago, so this "
                "one is close to the 10-business-day lost-parcel threshold.",
    },
    "NIM-77881122": {
        "email": "amir@example.com",
        "item": "Nimbus Pulse fitness band",
        "price": "PKR 6,400",
        "tier": "Standard",
        "placed": "14 days ago",
        "status": "DELIVERED",
        "returnable": "yes - delivered 8 days ago, inside the 30-day window",
        "note": "Delivered 8 days ago, so it is inside the 30-day return window.",
    },
    "NIM-55220147": {
        "email": "zoya@example.com",
        "item": "Nimbus Halo smart bulb, 2-pack",
        "price": "PKR 3,200",
        "tier": "Standard",
        "placed": "47 days ago",
        "status": "DELIVERED",
        "returnable": "no - delivered 41 days ago, PAST the 30-day window",
        "note": "Delivered 41 days ago, so it is PAST the 30-day return window. A "
                "fault would still be covered by the 12-month warranty.",
    },
    "NIM-90014455": {
        "email": "bilal@example.com",
        "item": "Nimbus Volt 65W charger",
        "price": "PKR 4,100",
        "tier": "Priority",
        "placed": "1 hour ago",
        "status": "PLACED",
        "returnable": "not yet - not delivered",
        "note": "Still PLACED, so it can be cancelled free and the address can "
                "still be changed.",
    },
    "NIM-31556780": {
        "email": "hina@example.com",
        "item": "Nimbus Vista indoor camera",
        "price": "PKR 11,500",
        "tier": "Express",
        "placed": "2 days ago",
        "status": "DISPATCHED",
        "returnable": "not yet - not delivered",
        "note": "Dispatched this morning, so it can NO LONGER be cancelled. It must "
                "be refused at the door or returned after delivery.",
    },
    "NIM-62003391": {
        "email": "omar@example.com",
        "item": "Nimbus Echo desk speaker",
        "price": "PKR 7,750",
        "tier": "Standard",
        "placed": "3 days ago",
        "status": "ON HOLD",
        "returnable": "not yet - not delivered",
        "note": "On hold for address verification. The customer must confirm their "
                "delivery address before it will move on.",
    },
}


def _render_orders() -> str:
    lines = [
        "DEMO ORDER BOOK (the only orders that exist; anything not listed here does",
        "not exist and you must say so rather than guessing):",
    ]
    for order_id, o in ORDERS.items():
        lines.append(
            f"- {order_id} | {o['email']} | {o['item']} | {o['price']} | "
            f"{o['tier']} | placed {o['placed']} | STATUS: {o['status']} | "
            f"RETURNABLE: {o['returnable']}"
        )
        lines.append(f"    {o['note']}")
    return "\n".join(lines)


ORDER_BOOK = _render_orders()

# The single string that gets injected into the system prompt.
KNOWLEDGE_BLOCK = "\n\n".join(
    [
        STORE_PROFILE,
        ORDER_LIFECYCLE,
        SHIPPING_POLICY,
        RETURNS_POLICY,
        PAYMENT_POLICY,
        IDENTIFIERS,
        ORDER_BOOK,
    ]
)
