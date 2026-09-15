# Nimbus Order Support — a local, CPU-only conversational agent

**NLP Assignment 1 — Conversational AI system**
Domain: **E-Commerce Order Support Assistant**

| Roll number | Name |
|---|---|
| 22i-2530 | Ahmed Javaid |
| 22i-2470 | Ammad Ashraf |

A customer-support chat agent for *Nimbus*, a fictional online electronics and
smart-home retailer. It runs a quantized open-weight LLM entirely on the local
CPU, streams responses token-by-token over a WebSocket, keeps conversation state
across turns, serves many sessions concurrently, and ships with a ChatGPT-style
web UI.

There are **no tools, no function calls, no agents and no RAG** anywhere in the
request path. Everything the assistant knows comes from two constants pasted
into the system prompt on every turn -- the Nimbus policy and a six-order demo
book -- plus what the customer said earlier in the same session. Nothing is
selected per query, there is no index, and no code fetches a record.

---

## Screenshots

Every conversation below is real output from the local model, captured by
[`scripts/make_screenshots.py`](scripts/make_screenshots.py): it drives the live
WebSocket API, then loads the resulting session by id in headless Chrome. None
of it is mocked up.

| | |
|---|---|
| ![Welcome, light](docs/screenshots/01-welcome-light.png) | ![Welcome, dark](docs/screenshots/02-welcome-dark.png) |
| **Light theme** — dotted background, suggestion cards | **Dark theme** — toggled in the header, persisted |

### A real order lookup

`NIM-40011234` with its matching email, so the assistant answers from the record:

![Order lookup](docs/screenshots/03-order-lookup.png)

### Refusing what it cannot know

An order reference that is not in the book. The assistant declines instead of
inventing a status:

![Unknown order](docs/screenshots/06-unknown-order.png)

### Staying in character

Off-topic and persona-override attempts. The deterministic guard *detects*
these, but the refusal is written by the model (`guard: steer:...`), so it stays
in voice — and the amber tint plus the chips make the mechanism visible:

![Off-topic and jailbreak](docs/screenshots/05-off-topic-guard.png)

### Policy arithmetic

An order delivered 41 days ago, so outside the 30-day window:

![Return past the window](docs/screenshots/04-return-past-window.png)

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [Phase I — Business case, policies and conversation flow](#3-phase-i--business-case-policies-and-conversation-flow)
4. [Phase II — Model selection and context memory](#4-phase-ii--model-selection-and-context-memory)
5. [Phase III — Conversation manager](#5-phase-iii--conversation-manager)
6. [Phase IV — API reference](#6-phase-iv--api-reference)
7. [Phase V — Web interface](#7-phase-v--web-interface)
8. [Phase VI — Benchmarks and evaluation](#8-phase-vi--benchmarks-and-evaluation)
9. [Testing](#9-testing)
9b. [Bonus claim](#9b-bonus-claim--ux--persona-polish)
10. [Known limitations](#10-known-limitations)
10b. [Viva prep](#10b-viva-prep)
11. [Submission](#11-submission)
12. [Repository layout](#12-repository-layout)

---

## 1. Quick start

### Prerequisites

| Requirement | Version used here |
|---|---|
| Python | 3.11+ |
| [Ollama](https://ollama.com/download) | 0.34.0 |
| RAM | ~4 GB free (3B model at Q4) |
| OS | Windows 11 (macOS/Linux work unchanged) |

### Setup

```bash
git clone <this-repo> && cd assignment-1
```

Create the virtual environment and install pinned dependencies:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`
> throughout.

> **Why pinned?** `fastapi==0.115.6` requires `starlette<0.42`. Installing
> `fastapi` unpinned picks up a `starlette` 1.x that breaks FastAPI at import
> time with `Router.__init__() got an unexpected keyword argument 'on_startup'`.

### Pull the model

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

Ollama runs as a background service after install. Verify it:

```bash
curl http://127.0.0.1:11434/api/version
```

### Run

```bash
.venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Open **<http://127.0.0.1:8000>**. Interactive API docs are at `/docs`.

On startup the server loads the model *and* pre-evaluates the shared system
prompt prefix, which is why the first real message is fast rather than taking
~13 seconds. Watch for `model warmup complete` in the log.

### Verify the install

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","engine":"ollama","model":"qwen2.5:3b-instruct-q4_K_M",
 "reachable":true,"model_present":true,"active_sessions":0,"version":"1.0.0"}
```

`"status":"degraded"` means Ollama is unreachable or the model was never pulled.

### Configuration

Every setting is an environment variable with a sane default — see
[`.env.example`](.env.example) for the full list. The ones worth knowing:

| Variable | Default | Purpose |
|---|---|---|
| `NIMBUS_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Which Ollama model to use |
| `NIMBUS_ENGINE` | `ollama` | `ollama`, `llamacpp`, or `mock` |
| `NIMBUS_FORCE_CPU` | `1` | Pins `num_gpu=0`; **required for this assignment** |
| `NIMBUS_CONTEXT_WINDOW` | `4096` | Runtime context size |
| `NIMBUS_HISTORY_TOKEN_BUDGET` | `1400` | Tokens of verbatim dialogue replayed |
| `NIMBUS_MAX_CONCURRENT_GENERATIONS` | `4` | Process-wide generation cap |
| `NIMBUS_GUARD_MODE` | `steer` | `steer` (model writes refusals), `reply` (canned), `off` |

To run the smaller/faster model instead:

```bash
NIMBUS_MODEL=qwen2.5:1.5b-instruct-q4_K_M .venv/Scripts/python.exe -m uvicorn app.api.main:app
```

---

## 2. Architecture

```mermaid
flowchart TB
    subgraph browser["Browser — frontend/"]
        UI["Chat UI<br/>index.html · app.js · styles.css"]
    end

    subgraph server["FastAPI process — app/"]
        WS["/ws/chat<br/>WebSocket · JSON frames"]
        REST["REST<br/>/health · /api/stats · /api/session"]

        subgraph cm["Conversation Manager — app/conversation/"]
            GUARD["Guard<br/>injection + off-domain<br/>(short-circuits the model)"]
            STATE["State tracker<br/>intent · slots · stage · topic switch"]
            MEM["Context memory<br/>pinned facts · window · rolling summary"]
            PROMPT["Prompt builder<br/>app/domain/prompts.py"]
            VERIFY["Output verification<br/>blocks invented order data"]
        end

        STORE[("Session store<br/>in-memory · TTL + LRU")]
        ENGINE["LLM engine interface<br/>app/llm/base.py"]
    end

    subgraph runtime["Local inference"]
        OLLAMA["Ollama daemon<br/>:11434 · num_gpu=0"]
        MODEL["qwen2.5 3B instruct<br/>Q4_K_M · CPU"]
    end

    UI <-->|"JSON frames<br/>token streaming"| WS
    UI -->|session history| REST
    WS --> GUARD
    GUARD -->|blocked| WS
    GUARD -->|allowed| STATE
    STATE --> MEM --> PROMPT --> ENGINE
    STATE <--> STORE
    ENGINE -->|"HTTP stream<br/>async, non-blocking"| OLLAMA --> MODEL
    MODEL -->|tokens| VERIFY --> WS

    style GUARD fill:#fde8e4,stroke:#c2371f,color:#1f2024
    style MEM fill:#eaf0ff,stroke:#2f6df6,color:#1f2024
    style VERIFY fill:#fde8e4,stroke:#c2371f,color:#1f2024
    style MODEL fill:#e6f5ec,stroke:#1f9d55,color:#1f2024
```

### Request path for one turn

```
user message
  │
  ├─ validate .......... type, empty, length cap (2000 chars)
  ├─ guard ............. regex pre-filter → canned reply, NO model call
  ├─ classify .......... support lane (intent) from surface cues
  ├─ extract slots ..... order ref / email / RMA → pinned facts
  ├─ topic switch ...... park the old lane, or resume a parked one
  ├─ advance stage ..... the Nimbus stage machine
  ├─ build prompt ...... stable prefix ‖ summary ‖ facts ‖ stage ‖ history ‖ msg
  ├─ stream ............ async HTTP → Ollama → tokens → WebSocket
  ├─ scrub ............. strip any leaked prompt scaffolding
  ├─ verify ............ block invented order state; replace if found
  ├─ commit turn ....... append to session history
  └─ refresh summary ... only if history was evicted, AFTER the answer is shown
```

### One turn, end to end

The component diagram shows *what exists*; this shows *what happens*, including
where the two safety mechanisms sit and why the customer sees text before the
model has finished thinking.

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant WS as /ws/chat
    participant CM as Conversation Manager
    participant OL as Ollama (CPU)

    U->>WS: {"type":"chat","message":"where is my order?"}
    WS->>CM: stream_turn(session, message)

    rect rgb(253, 232, 228)
        Note over CM: validate · guard (regex)
        CM->>CM: off-domain? inject a steer instruction
    end

    rect rgb(234, 240, 255)
        Note over CM: classify lane · pin order ref + email<br/>advance stage · plan context window
    end

    CM->>OL: system (cached prefix) + history + new turn
    Note over OL: prompt eval ~64 tok/s<br/>decode ~12 tok/s

    WS-->>U: {"type":"start","stage":"order_identification"}
    loop every token
        OL-->>CM: delta
        CM-->>WS: delta
        WS-->>U: {"type":"token","text":"…"}
    end

    rect rgb(253, 232, 228)
        Note over CM: scrub scaffolding · verify no invented order state
        CM-->>WS: {"type":"correction"} only if a fabrication was caught
        WS-->>U: replace the message
    end

    WS-->>U: {"type":"done","stats":{…},"context":{…}}
    Note over CM: rolling summary refreshes here —<br/>after the answer is already on screen
```

Two things worth noticing. The `done` frame is sent **before** the rolling
summary runs, so compression never costs the customer latency. And the
`correction` frame exists because tokens are streamed as they arrive — by the
time verification can inspect a complete reply, the customer has already read
it, so the fix is a replacement rather than a block.

### Why the layers are split this way

- **`app/domain/`** holds everything Nimbus-specific — the policy text, the
  stage machine, the guard, the prompt builder. Changing the business to a car
  rental assistant means editing this folder and nothing else.
- **`app/conversation/`** is domain-agnostic machinery: sessions, context
  budgeting, turn orchestration.
- **`app/llm/`** is a four-method interface (`stream_chat`, `complete`,
  `health`, `warmup`) with three implementations. That is why the test suite can
  run the entire system against a deterministic in-process double in 1.5
  seconds, and why swapping Ollama for llama.cpp is a one-variable change.
- **`app/api/`** owns only transport concerns: frame parsing, error mapping,
  concurrency limits.

---

## 3. Phase I — Business case, policies and conversation flow

### 3.1 The use case

**Nimbus** sells consumer electronics and smart-home gear online: audio,
wearables, smart home devices, laptop/phone accessories, and power products.
Its support inbox is dominated by four questions — *where is my order*, *can I
return this*, *it broke, what now*, and *what are your shipping/payment rules* —
and the first-line answers to all four are pure policy lookups against facts the
customer already has. That is exactly the shape of problem a prompt-only
assistant can genuinely solve, which is why it was chosen.

**Ava**, the assistant, is an Order Support Specialist: calm, concise, and
practical. She is explicitly *not* a salesperson and not a chirpy brand mascot.

### 3.2 What the assistant will and will not do

| Will | Will not |
|---|---|
| Look up any of the six demo orders, once the reference **and** email match | Say anything about an order that is not in the book, or whose email does not match |
| Explain shipping tiers, costs, cut-offs and transit times | Invent a delivery date, a carrier, or a named day |
| Explain the warranty route for a faulty item | Issue an RMA number or a refund itself |
| Collect and confirm an order reference and account email | Grant a refund, discount, extension or goodwill credit |
| Explain cancellation and address-change windows | Answer anything outside Nimbus order support |
| Escalate to a human at `support@nimbus.example` | Adopt a different persona or reveal its instructions |

The critical constraint is the second row. The assistant has **no order
database** — that would require tools or retrieval, which the assignment
forbids. Rather than pretending otherwise, the design makes the limitation
explicit and still useful: Ava collects and validates identifiers, applies the
policy to what the customer tells her, and states the concrete next step. The UI
says so in the welcome screen, so the user is never misled.

This turned out to be the hardest behaviour to enforce, and
[§8.2](#82-correctness-evaluation) documents what it took.

### 3.3 Conversation flow design

Eight named stages. They are Nimbus-specific rather than a generic
greet → collect → close skeleton: `ORDER_IDENTIFICATION` exists because *every*
order-touching lane at Nimbus needs a reference plus the account email before
anything can be actioned, and `POLICY_RESOLUTION` exists because the deliverable
of a Nimbus support turn is a **policy verdict** — eligible or not, the exact
window, the fee, the single next step.

```mermaid
stateDiagram-v2
    [*] --> GREETING
    GREETING --> INTENT_TRIAGE: customer states a need
    INTENT_TRIAGE --> POLICY_RESOLUTION: product / policy question<br/>(no identity needed)
    INTENT_TRIAGE --> ORDER_IDENTIFICATION: order · cancel · return · warranty

    ORDER_IDENTIFICATION --> ORDER_IDENTIFICATION: reference or email still missing
    ORDER_IDENTIFICATION --> ISSUE_DETAIL: identity complete,<br/>lane detail missing
    ORDER_IDENTIFICATION --> POLICY_RESOLUTION: policy already decides it<br/>(e.g. already dispatched)
    ISSUE_DETAIL --> POLICY_RESOLUTION: all slots filled
    POLICY_RESOLUTION --> CONFIRMATION: read back + confirm
    CONFIRMATION --> CLOSING
    POLICY_RESOLUTION --> CLOSING: "thanks, that's all"
    CLOSING --> [*]

    INTENT_TRIAGE --> OUT_OF_SCOPE: off-domain / persona attack
    ORDER_IDENTIFICATION --> OUT_OF_SCOPE
    POLICY_RESOLUTION --> OUT_OF_SCOPE
    OUT_OF_SCOPE --> INTENT_TRIAGE: customer returns to a Nimbus topic
```

| Stage | What Ava does |
|---|---|
| `GREETING` | One-line intro, offers the three support lanes. Does **not** ask for an order reference yet. |
| `INTENT_TRIAGE` | Works out the lane; at most one clarifying question. |
| `ORDER_IDENTIFICATION` | Checks first whether policy already settles it; otherwise asks for reference + email **in one message**, and only for the missing piece. |
| `ISSUE_DETAIL` | Gathers lane-specific facts (tracking state, delivery date, reason, fault). |
| `POLICY_RESOLUTION` | States the verdict with exact numbers from policy. |
| `CONFIRMATION` | Reads back collected details, asks the customer to confirm. |
| `CLOSING` | Says what happens next and by when. |
| `OUT_OF_SCOPE` | Declines in one or two sentences, names what it does handle, steers back. |

Each lane also carries its own directive on top of the stage
(`LANE_DIRECTIVE` in [`app/domain/policy.py`](app/domain/policy.py)). The
order-status one is load-bearing — see [§8.2](#82-correctness-evaluation).

### 3.4 Handling a topic change mid-conversation

Real customers switch topics. The manager treats this as first-class state
rather than letting it corrupt the flow:

1. **Detect.** Each turn is re-classified. A new lane while the previous lane
   still has unfilled required slots is a *switch*.
2. **Park, don't discard.** The previous lane is stored in
   `session.suspended_intent`, and the identifiers already collected stay pinned
   in `session.facts`.
3. **Instruct.** A turn-specific note tells Ava to acknowledge the switch in at
   most half a sentence, handle the new topic, and offer to return to the parked
   one at the end. No re-greeting.
4. **Resume.** When the customer comes back to the parked lane, a *different*
   note fires: pick it up where it left off, do not start over. This check runs
   **before** the switch check, because returning to a parked lane is also a
   change of intent and would otherwise be handled as a fresh switch.

Real transcript of exactly this — see Dialogue 2 in
[`docs/example-dialogues.md`](docs/example-dialogues.md).

### 3.5 Refusing irrelevant queries

Three layers, cheapest first:

**Layer 1 — deterministic guard (no model call).**
[`guard()`](app/domain/policy.py) runs regex families for persona-override
attempts (*ignore previous instructions*, *you are now*, *what is your system
prompt*, *developer mode*, *pretend to be…*) and for clearly off-domain topics
(coding help, homework/essays, medical/legal/financial advice, politics, general
knowledge, other retailers' orders). A hit returns a canned, in-character
deflection in **~1 ms**, with the model never invoked.

Three properties make this worth having over prompt-only refusal:
it is **free** (no tokens), **instant** (no CPU decode), and **perfectly
consistent** (a 1.5B model's refusals are not). The protocol marks these turns
with a `guarded` field so the UI can tint them and the grader can see the
mechanism working.

It is deliberately **conservative**: anything it is not confident about passes
through to the model. Legitimate-but-superficially-similar messages are
explicitly tested — *"is the Nimbus smart plug compatible with my home
automation setup?"* and *"my promo code did not apply"* both pass.

**Guard modes.** There is a genuine tension between wanting deterministic
*detection* and the assignment's rule that *"every response must come from
prompt orchestration and conversational memory alone"* — a hardcoded refusal is
not model-generated. `NIMBUS_GUARD_MODE` resolves it explicitly:

| Mode | Detection | Who writes the reply | Cost |
|---|---|---|---|
| **`steer`** (default) | regex, deterministic | **the model**, from an injected instruction | one normal generation |
| `reply` | regex, deterministic | a canned string, model never called | ~1 ms |
| `off` | none | the model, guided only by the system prompt | one normal generation |

The default is **`steer`**: the regex decides *that* a message is off-domain,
and the model decides *how* to decline. That keeps every customer-visible word
model-generated — fully inside the constraint — while still getting consistent,
free detection. `reply` remains available when latency matters more than having
the text generated, and `off` demonstrates that the system prompt alone still
refuses (it does, just less reliably).

**Layer 2 — system prompt.** Rule 1 of `HARD_RULES` scopes the assistant, and
`FINAL_GUARDRAIL` restates the two most-violated rules in the last position of
the prompt where attention is strongest.

**Layer 3 — output scrubbing.** `scrub()` strips any prompt scaffolding the
model echoes (`CURRENT STAGE:`, `Agent:`, a leaked heading and its bullets)
before it reaches the transcript or the history.

**Layer 4 — output verification.** The one thing that must never reach a
customer is a *fabricated* order status.
[`app/domain/verification.py`](app/domain/verification.py) checks every reply in
an order-touching lane, and since the order book exists the verdict depends on
identity:

| Situation | Verdict |
|---|---|
| Reference in the book **and** email matches | May state that record. Only a *contradicting* status, or a wrong return verdict, is caught |
| Reference unknown, email mismatched, or no reference | Any status claim or lookup claim is invented — **blocked** |
| Denial ("I cannot find that order") | Never fabrication, in any case |
| Conditional or definitional ("once an order is dispatched…") | Correct policy talk, always allowed |
| A named day or date | No record carries one — **blocked** either way |

A blocked reply is replaced in the transcript *and in the stored history*, so it
cannot be replayed next turn and become context the model builds on. The client
gets a `correction` frame and swaps the message.

**Testing this against the real model found five genuine defects**, each worth
knowing about because they are the kind that hide behind a green test suite:

1. Patterns anchored on *"your order"* missed *"**this** order is currently in the
   DELIVERED side state"* — a flat contradiction of the record.
2. `state` and `status` were in the allow-list to spare definitions, and became a
   bypass: the word "state" alone excused a contradicting sentence.
3. Greedy regex captured the *furthest* state, so *"in transit and should be
   delivered in a few days"* was flagged as contradicting itself.
4. Return eligibility is arithmetic the state checks are blind to: the model
   wrote *"you are within the 30-day return window since it was delivered 41 days
   ago"*. Records now carry an explicit `returnable` field so the contradiction
   is checkable without parsing English.
5. Blocking *"I can confirm there **isn't** an order"* replaced a correct denial
   with a fallback.

Both directions are pinned in
[`tests/test_fabrication_detector.py`](tests/test_fabrication_detector.py): a
detector that fires on correct answers is worse than none, because it silently
degrades working behaviour.

### 3.7 Demo order book — what you can actually test

Six orders live permanently in the system prompt. Use the reference **with its
matching email** to get a real answer; use anything else to see the assistant
refuse rather than guess.

| Order | Email | Item | Status | Returnable |
|---|---|---|---|---|
| `NIM-40011234` | `sara.k@example.com` | Nimbus Aura 2 wireless earbuds | **IN TRANSIT** | not yet - not delivered |
| `NIM-77881122` | `amir@example.com` | Nimbus Pulse fitness band | **DELIVERED** | yes - delivered 8 days ago, inside the 30-day window |
| `NIM-55220147` | `zoya@example.com` | Nimbus Halo smart bulb, 2-pack | **DELIVERED** | no - delivered 41 days ago, PAST the 30-day window |
| `NIM-90014455` | `bilal@example.com` | Nimbus Volt 65W charger | **PLACED** | not yet - not delivered |
| `NIM-31556780` | `hina@example.com` | Nimbus Vista indoor camera | **DISPATCHED** | not yet - not delivered |
| `NIM-62003391` | `omar@example.com` | Nimbus Echo desk speaker | **ON HOLD** | not yet - not delivered |

Between them they cover every policy branch: in transit, delivered-and-returnable,
delivered-past-window, cancellable, too-late-to-cancel, and on-hold.

**Is this RAG?** No. The whole book is rendered into the system prompt on every
turn, exactly like the policy text — nothing is selected in response to the
question, there is no index, and no code fetches a record to answer a query. The
model reads what is already in its context. It *would* become retrieval the
moment the book grew large enough that we had to choose which rows to include,
which is precisely why it is capped at six and lives in a constant.

**Identity is enforced.** A reference alone proves nothing — anyone could guess
one — so the email must match too. Try `NIM-40011234` with `attacker@example.com`
and the assistant declines to reveal the status.

### 3.8 Example dialogues

Three full transcripts — happy path, return with a mid-conversation topic
switch, and adversarial probing — are in
**[`docs/example-dialogues.md`](docs/example-dialogues.md)**.

They are **real captured output**, produced by
[`scripts/make_transcripts.py`](scripts/make_transcripts.py) against the running
local model, annotated with the manager's actual per-turn stage and measured
time-to-first-token. They are not hand-written illustrations. Regenerate them
with:

```bash
.venv/Scripts/python.exe scripts/make_transcripts.py > docs/example-dialogues.md
```

---

## 4. Phase II — Model selection and context memory

### 4.1 Model choice

**Selected: `qwen2.5:3b-instruct-q4_K_M`** (3.1B parameters, Q4_K_M, 1.9 GB).

Two candidates were pulled and run through the *same* 21-check evaluation suite
and the same latency benchmark on this machine, rather than picking on
reputation:

| | Qwen2.5 1.5B Q4_K_M | **Qwen2.5 3B Q4_K_M** |
|---|---|---|
| Disk / RAM | 0.99 GB | 1.93 GB |
| Evaluation score | see [§8.2](#82-correctness-evaluation) | see [§8.2](#82-correctness-evaluation) |
| Decode throughput | see [§8.1](#81-latency-benchmarks) | see [§8.1](#81-latency-benchmarks) |

Why Qwen2.5 Instruct as the family:

- **Instruction-following at small scale.** This system leans entirely on prompt
  adherence — an eight-stage machine, a seven-rule policy block and per-lane
  directives, with no tools to fall back on. Qwen2.5's instruct tuning holds a
  long structured system prompt better than comparable-size alternatives.
- **Q4_K_M specifically.** The assignment requires Q4. K-quants beat legacy
  `Q4_0` at the same bit-width, and `_M` keeps higher precision on the attention
  and feed-forward tensors that matter most for instruction adherence.
- **Fits the box.** 1.9 GB of weights plus a 4k KV cache runs comfortably in
  ~4 GB, leaving headroom for concurrent sessions.
- **Open weights, Apache-2.0, runs offline.** No cloud API anywhere.

**CPU-only is enforced in code, not by convention.** Ollama silently offloads to
any discrete GPU it finds — on this machine it initially ran *100% on an
RTX 3080* at 188 tok/s, which would have violated the assignment. The engine now
pins `num_gpu: 0` on **every request**, so the guarantee does not depend on how
the daemon was launched. `ollama ps` confirms `100% CPU`.

### 4.2 Context memory management

CPU prompt evaluation is **linear in prompt length**. An unbounded history turns
a sub-second first token into a ten-second one by turn twenty — measured, see
[§8.1](#81-latency-benchmarks). The scheme is a **three-tier memory**:

```
┌─ TIER 1 · PINNED FACTS ─────────────────── ~20 tokens, never evicted ─┐
│  order_id: NIM-40011234   email: sara.k@example.com   rma: …          │
│  Regex-extracted from user turns, re-injected into the system prompt  │
│  every turn. Survives eviction entirely.                              │
├─ TIER 2 · VERBATIM WINDOW ──────────── ≤1400 tokens, newest-first ────┤
│  turn n-3 │ turn n-2 │ turn n-1 │ turn n                              │
│  Walk backwards, accumulate until the budget is spent.                │
│  Floor: the newest 2 exchanges are kept even if that overruns.        │
├─ TIER 3 · ROLLING SUMMARY ─────────────────────── ≤160 tokens ────────┤
│  Everything evicted from tier 2, compressed by the same local model   │
│  at temperature 0. Rewritten, not appended, so it cannot grow.        │
└───────────────────────────────────────────────────────────────────────┘
```

**Why three tiers and not just a sliding window.** A plain window has one
catastrophic failure in this domain: the customer gives their order reference on
turn 2, and on turn 9 the assistant asks for it again because turn 2 fell out.
Tier 1 fixes precisely that for ~20 tokens — high-precision, format-checkable
values (`NIM-\d{8}`, an email, `RMA-\d{7}`) are pinned and replayed forever.
Fuzzy slots (item, reason, fault) are deliberately *not* extracted, because
guessing them would put words in the customer's mouth.

**Budget arithmetic**, against a 4096-token window:

| Component | Tokens |
|---|---:|
| System prompt (persona + policy + rules + style + guardrail) | ~1,650 |
| Rolling summary + pinned facts | ~180 |
| Verbatim history (`NIMBUS_HISTORY_TOKEN_BUDGET`) | 1,400 |
| Reserved for generation (`NIMBUS_MAX_OUTPUT_TOKENS`) | 320 |
| **Total worst case** | **~3,550 / 4,096** |

**The summariser never costs the customer latency.** It runs *after* the `done`
frame has been sent, so the answer is already fully on screen; the result
arrives as a later `context` frame. A watermark (`summarised_upto`) means each
evicted turn is folded in exactly once — without it the same early turns get
re-summarised on every later eviction and the summary drifts. If the summariser
call fails, the previous summary is kept rather than dropped.

**Token counting is estimated, and the estimate is measured.** Ollama exposes no
tokenizer endpoint, and pulling in a HuggingFace tokenizer for a number that
only needs ~10% accuracy is not worth the dependency. `CHARS_PER_TOKEN = 3.6`
is validated against Ollama's own `prompt_eval_count` by
`python scripts/benchmark.py --calibrate` — results in
[§8.1](#81-latency-benchmarks).

**Prompt block order is a performance decision, not cosmetics.** llama.cpp
reuses its KV cache for the longest common prefix between consecutive prompts.
Putting the ~1,650-token persona + policy block first, unchanged for the whole
session, means it is prompt-evaluated **once per session** instead of once per
turn. That is the single biggest TTFT win in the system — see the cold-vs-warm
gap in [§8.1](#81-latency-benchmarks). Volatile blocks (summary, facts, stage,
notes) all sit *after* it, which also puts them where a small model's attention
is strongest.

---

## 5. Phase III — Conversation manager

[`app/conversation/manager.py`](app/conversation/manager.py). One public
coroutine, `stream_turn(session, message)`, yields `TurnEvent`s until the turn
is done.

**Dialogue history.** `Session.turns` is the full ordered transcript; what is
*replayed* is decided per-turn by `plan_window()`. Nothing is deleted — eviction
only means "not replayed verbatim", so `/api/session/{id}` can still return the
complete conversation to the UI.

**Turn-taking.** One generation per session at a time, enforced by
`asyncio.Lock`. A second send while the first is running is **rejected with a
`busy` error rather than queued** — a queued turn would answer a question the
customer can no longer see in context.

**Structured system prompts.** Assembled in a fixed order by
[`build_system_prompt()`](app/domain/prompts.py) from seven blocks. Stable ones
first (KV-cache prefix), volatile ones last (attention recency).

**Output verification.** After scrubbing, a reply in an order-touching lane is
checked for invented order state and replaced if found (§3.5, layer 4). The
replacement is what gets stored, so a fabrication can never become context for
the next turn.

**Faithfulness across turns** comes from the three-tier memory in §4.2, plus a
`CONFIRMED DETAILS … do not ask for them again` block, and is regression-tested:
`test_facts_survive_history_eviction` buries an order reference under 30 turns of
filler and asserts it is still in the prompt.

---

## 6. Phase IV — API reference

### WebSocket `/ws/chat`

Optional `?session_id=<id>` resumes an existing session (used on page refresh).

**Client → server**

| Frame | Shape |
|---|---|
| `chat` | `{"type":"chat","message":"where is my order?"}` |
| `reset` | `{"type":"reset"}` |
| `ping` | `{"type":"ping"}` |

**Server → client**

| Frame | Shape | When |
|---|---|---|
| `ready` | `{"type":"ready","session_id":…,"stage":…,"engine":…,"model":…,"turn_count":0}` | On connect |
| `start` | `{"type":"start","stage":"order_identification","intent":"order_status","guarded":""}` | Turn accepted |
| `token` | `{"type":"token","text":"Express "}` | Per delta |
| `done` | `{"type":"done","stage":…,"intent":…,"stats":{…},"context":{…}}` | Turn complete |
| `correction` | `{"type":"correction","text":…,"reason":"fabricated_order_data"}` | Output verification replaced the reply |
| `context` | `{"type":"context","context":{"summary":…}}` | After a rolling-summary refresh |
| `reset_ok` | `{"type":"reset_ok","session_id":…}` | After `reset` |
| `pong` | `{"type":"pong"}` | After `ping` |
| `error` | `{"type":"error","code":"busy","message":…,"fatal":false}` | Any recoverable failure |

`done.stats` carries `ttft_ms`, `total_ms`, `prompt_tokens`,
`completion_tokens`, `decode_tps`, `overall_tps`.
`done.context` carries `turns_verbatim`, `turns_total`, `turns_evicted_total`,
`history_tokens`, `prompt_tokens_est`, `has_summary`, `pinned_facts` — this is
what the UI's **Telemetry** toggle displays.

**Error codes**

| Code | Meaning | Socket stays open |
|---|---|---|
| `malformed_json` | Frame was not JSON | yes |
| `malformed_frame` | JSON but not an object | yes |
| `invalid_frame` | Unknown type, or message over the cap | yes |
| `missing_message` | `chat` frame with no `message` | yes |
| `empty_message` | Message was blank | yes |
| `message_too_long` | Over `NIMBUS_MAX_MESSAGE_CHARS` | yes |
| `busy` | A generation is already running for this session | yes |
| `engine_unreachable` | Ollama is down | yes |
| `model_not_found` | Model not pulled | yes |
| `engine_timeout` | Generation exceeded the timeout | yes |
| `upstream_error` | Runtime returned an error | yes |
| `internal_error` | Unexpected exception | yes (`fatal` if not) |

**Every one of these leaves the connection usable.** There is no code path
where a bad frame closes the socket — tested in
[`tests/test_api.py`](tests/test_api.py).

### REST

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether the model is actually present |
| `GET` | `/api/stats` | Active sessions, in-flight generations, limits |
| `POST` | `/api/session` | Create a session |
| `GET` | `/api/session/{id}` | Full history, stage, pinned facts, summary |
| `DELETE` | `/api/session/{id}` | Clear history, keep the id |
| `GET` | `/docs` | Swagger UI |

### Concurrency

Everything on the request path is `async`; the only CPU-bound work happens
inside the Ollama process, reached over non-blocking HTTP. Two limits:

- a process-wide **semaphore** (`NIMBUS_MAX_CONCURRENT_GENERATIONS`, default 4)
  — six cores running eight decodes at once makes *everyone* slow;
- a per-session **lock** — one tab, one in-flight generation.

`test_one_slow_generation_does_not_block_other_requests` asserts a REST call is
served while a generation is mid-stream.

---

## 7. Phase V — Web interface

Plain HTML/CSS/JS, no framework and no build step —
[`frontend/`](frontend/) is three files served directly by FastAPI.

- **Live streaming** — tokens appended as text nodes (not `innerHTML`
  re-renders), with a blinking cursor while generating.
- **Full history**, with the conversation stage shown as a chip under each reply,
  and restored on refresh: the session already survived on the server, so the
  transcript now repaints instead of coming back to an empty thread.
- **Deep links** - `?session_id=...` opens a specific conversation and
  `?theme=light|dark` forces a theme, which is how the screenshots above are
  captured reproducibly.
- **New chat** resets the session over the socket.
- **Telemetry toggle** — per-reply TTFT, tok/s, prompt/output tokens, and the
  live context-window state (`ctx 4/17 turns`, `evicted 13`, `summary on`,
  `pinned: email, order_id`). This makes the memory scheme in §4.2 visible while
  you use it.
- **Guarded replies are tinted** and labelled with the guard reason; a reply
  replaced by output verification is tinted too and labelled
  `corrected: invented order data`, so the mechanism is visible, not silent.
- **Resilience** — auto-reconnect with exponential backoff, 25 s heartbeat,
  session id in `sessionStorage` so a refresh resumes the conversation, and a
  stop control that aborts an in-flight generation.
- **Honest empty state** — the welcome screen says Ava has no live access to the
  order system, so the user is never misled about what it can do.
- **Light and dark**, with an explicit toggle that persists in `localStorage`
  and otherwise follows the OS. A dotted background grid and a soft accent
  wash keep the page from reading as a flat slab.
- Responsive to ~375 px, keyboard shortcuts, and `prefers-reduced-motion`
  respected.

---

## 8. Phase VI — Benchmarks and evaluation

Hardware for every number below:

| | |
|---|---|
| CPU | AMD Ryzen 5 3600 — 6 cores / 12 threads |
| RAM | 16 GB |
| OS | Windows 11 Pro (build 26200) |
| Runtime | Ollama 0.34.0, `num_gpu=0` (GPU explicitly disabled) |
| Context | 4096 tokens |

> A GeForce RTX 3080 is present in this machine but is **disabled for all
> measurements** — see §4.1. For reference, the same model on that GPU reached
> 188 tok/s versus ~26 tok/s on CPU; every number in this section is the CPU
> figure.

Reproduce everything below with:

```bash
.venv/Scripts/python.exe scripts/benchmark.py --all --rounds 3 --concurrent 4
```

Raw output: [`docs/benchmarks-3b.md`](docs/benchmarks-3b.md),
[`docs/benchmarks-1.5b.md`](docs/benchmarks-1.5b.md) and
[`docs/benchmark-conversation.md`](docs/benchmark-conversation.md).

### 8.1 Latency benchmarks

> **Read the multi-turn table first.** Two latency numbers appear below and they
> differ by 30x. The single-turn figure re-sends an identical prompt, so the
> runtime's KV cache covers nearly all of it — that is a best case a real
> conversation never reaches. The multi-turn figure runs an actual 8-turn
> conversation through the conversation manager and is what a user feels.
> Publishing only the first would flatter this system by an order of magnitude.

#### Realistic multi-turn conversation (the number that counts)

`python scripts/benchmark.py --conversation` — one 8-turn support conversation
through the real conversation manager:

| Metric | Before optimisation | **After** |
|---|---:|---:|
| TTFT median | 16.1 s | **7.9 s** |
| TTFT, steady-state turns | 15.1 s mean | **6.5 s mean** |
| Total response mean | 26.5 s | 25.4 s |

**What changed, and why it worked.** The original layout appended the stage
directive, pinned facts and per-turn notes to the *system prompt*. Since the
dialogue history sits after the system message, that divergence at ~token 1,650
put **every history turn on the far side of the cache boundary** — so each turn
re-read the whole conversation. The evidence was that TTFT tracked *total*
prompt tokens (1,786 → 2,776 → 5 s → 22 s) rather than new ones.

Two fixes:

1. **Moved all turn-specific context out of the system prompt** and into the
   final user message. The system prompt is now byte-identical every turn and
   history entries are the customer's raw words, so the cacheable prefix grows
   by appending instead of being invalidated:

   ```
   [ system: fixed ] [ turn 1 ] [ turn 2 ] … [ new turn + context ]
   \________________ cached, grows by appending ________________/  ^ only this
   ```

   `test_system_prompt_is_identical_across_turns` and
   `test_history_is_replayed_as_raw_customer_words` stop this regressing.

2. **Rewrote the stage and lane directives from prose to clipped imperatives.**
   They ride past the cached prefix, so unlike the policy block every token is
   re-evaluated every turn — 206 tokens now, down from ~350.

TTFT also stopped climbing with conversation length, which was the real defect.
Two outlier turns in the published run (29.9 s and 22.3 s) are Ollama reloading
an evicted model, not prompt cost; the median is the honest figure.

**Still not fast.** ~8 s to first token on six CPU cores is a demo-grade number,
not a production one. Remaining levers, in order: shrink the 930-token policy
block, or use a GPU (188 tok/s on the RTX 3080 versus 12 on CPU) — which this
assignment rules out.

#### Single-turn latency (repeated identical prompt — best case)

18 warm generations, 3 rounds over 6 prompts, same system prompt each time:

| Metric | Mean | Median | p95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| Time to first token | 518 ms | 493 ms | 738 ms | 347 ms | 776 ms |
| Total response time | 6.96 s | 6.20 s | 12.01 s | 2.13 s | 16.25 s |
| Decode throughput | 12.1 tok/s | 12.0 | 13.6 | 9.2 | 13.7 |
| Completion length | 75 tok | 68 | 129 | 24 | 154 |

This isolates **decode** speed (12.1 tok/s) cleanly, which is its real value —
decode is prompt-independent, so this number does transfer. The 518 ms TTFT does
not.

**Uncached first token: 27.49 s** — the full 1,747-token prompt evaluated with
nothing cached, at **64 tok/s prompt throughput**. The startup warmup pays this
once so the first customer does not.

**Model comparison, measured on identical prompts and hardware:**

| | Qwen2.5 1.5B Q4_K_M | **Qwen2.5 3B Q4_K_M** (selected) |
|---|---:|---:|
| TTFT, single-turn best case (mean) | 990 ms | **518 ms** |
| TTFT, single-turn best case (p95) | 2,288 ms | **738 ms** |
| Total response (mean) | **4.45 s** | 6.96 s |
| Decode throughput | **21.4 tok/s** | 12.1 tok/s |
| Uncached first token | **17.07 s** | 27.49 s |
| Prompt-eval throughput | **135 tok/s** | 64 tok/s |
| Evaluation score, raw model output | 90%, 86% (**mean 88%**) | 95%, 90% (**mean 92.5%**) |
| Disk / RAM | **0.99 GB** | 1.93 GB |

**The 3B was chosen because the quality gap is on the failures that matter and
the latency cost is affordable.** It is ~1.8× slower to decode, but 7 seconds
for a complete support reply is acceptable for chat, streaming hides most of it
(first token in half a second), and the 1.5B's p95 TTFT is actually *worse*
(2.3 s vs 0.74 s) because it evicts its prefix cache more often. Concretely, the
1.5B repeatedly asserted things the policy block explicitly contradicts —
inventing "international shipments typically take 5–7 business days" when Nimbus
offers no international shipping. Set `NIMBUS_MODEL=qwen2.5:1.5b-instruct-q4_K_M`
to trade that accuracy back for ~2× the speed.

**Prompt length vs TTFT** — the measurement the context policy exists for:

| Extra history turns | Prompt tokens | TTFT, cache **miss** | TTFT, cache **hit** | Decode |
|---:|---:|---:|---:|---:|
| 0 | 1,743 | 26.8 s | 136 ms | 9.2 tok/s |
| 4 | 1,959 | 31.3 s | 125 ms | 11.3 tok/s |
| 8 | 2,175 | 33.7 s | 129 ms | 11.1 tok/s |
| 16 | 2,613 | 42.2 s | 139 ms | 11.1 tok/s |
| 24 | 3,053 | 50.4 s | 141 ms | 10.7 tok/s |

Two different facts, and both matter:

- The **miss** column is linear — about **18 ms per prompt token**. This is what
  a prompt actually costs when the cache cannot help: first turn, after a
  restart, or when concurrent sessions recycle the runtime's cache slots. Left
  unbounded, history would push this past a minute *and* overflow the 4,096-token
  window outright.
- The **hit** column is flat (~130 ms), because the stable prompt prefix means
  only newly-appended tokens need evaluating. That flatness is *earned* by the
  block ordering in §4.2, not free.

> An earlier version of this benchmark only measured the hit case and made the
> context policy look unnecessary. Forcing a genuine cache miss is what showed
> the real cost.

**Token estimator calibration** (`--calibrate`):
`CHARS_PER_TOKEN = 4.0` gives a **+3.5% mean error** (max 3.7%) against Ollama's
reported `prompt_eval_count`. The first value tried, 3.6, over-estimated by
+14.9%; the implied best fit is 4.14, and 4.0 is used so the estimator stays
slightly conservative — errors shorten the prompt rather than overflow the
window.

### 8.2 Correctness evaluation

```bash
.venv/Scripts/python.exe scripts/evaluate.py -v
```

[`scripts/evaluate.py`](scripts/evaluate.py) runs **10 scripted conversations /
21 assertions** through the full conversation manager against the real model.
These are heuristic string checks, not a model grading a model — stated plainly
rather than dressed up as a formal eval.

| Scenario group | What it proves |
|---|---|
| Happy path, late delivery (3 turns) | Asks for identification; never narrates the parcel |
| Returns window + opened-audio exception | Quotes 30 days, applies the hygiene rule |
| Cancellation after dispatch | States the cut-off instead of softening it |
| Memory across a 4-turn detour | Does not re-ask for a reference given on turn 1 |
| Off-topic: general knowledge, coding | Declines, answers neither |
| Adversarial: persona override, prompt extraction | Persona holds, prompt does not leak |
| Adversarial: pressure for an exception | Holds the window or escalates; approves nothing |
| Adversarial: demand for a live lookup | Claims no visibility |
| Topic switch mid-flow | Follows without re-greeting |

**Result: 21/21 (100%) of checks pass on the delivered output**, with output
verification intercepting **2 fabricated replies** in the run.

Those two numbers have to be read together, and the script prints both:

```
21/21 checks passed (100%) across 10 scenarios
output verification fired 2 time(s) -- replies the model fabricated that never reached the customer
    intercepted: 'Your order is currently in transit'
    intercepted: 'your order was dispatched'
```

The 100% is what a **customer** experiences. The "fired 2 times" is what the
**model** did, and it is the number that would quietly rot if it were hidden
inside the pass rate: the 3B model still invents order state roughly twice per
ten conversations, and the safety net is what stops it. Reporting only the 100%
would be a real misrepresentation of how much of that score the prompt earns
versus how much the verifier carries.

For reference, before output verification existed the same suite scored **90%
(3B)** and **88% (1.5B)** on raw model output.

> **The measurement itself needed fixing twice**, which is worth recording.
> First, the original fabrication patterns missed the real phrasing *"your order
> is now in the IN TRANSIT phase"* — a detector that misses what it exists to
> catch is worse than none. Second, once verification shipped, the evaluation
> was still scoring the *pre-correction* token stream, so it reported failures
> the system had already handled. Both are now fixed, and the evaluation calls
> the exact same `find_fabrication()` the runtime does, so it can never be
> stricter or looser than what actually ships.

**What the evaluation actually caught**, and what fixing it required — this is
the part worth reading:

1. **Fabricated order data.** Asked "where is my parcel", the assistant replied
   *"Your parcel has been dispatched on Tuesday, October 5th, from our Lahore
   warehouse"* — inventing a date, a status and a location. Rule 2 of
   `HARD_RULES` already forbade exactly this, but it sits ~1,000 tokens from the
   end of the prompt and a small model underweights it. **Three changes fixed
   it:** a short `FINAL_GUARDRAIL` block restating only the two most-violated
   rules in the last prompt position; a per-turn note injected *only* in
   order-touching lanes (a note that fires every turn stops being read); and a
   lane-specific directive telling the model what a *correct* answer looks like
   when you cannot see the order.
2. **A policy verdict skipped for identification.** "My order was dispatched
   this morning, can I cancel?" made the assistant ask for an order reference
   for an action the policy makes impossible. That was a genuine bug in the
   stage machine, not the model: `ORDER_IDENTIFICATION` now checks whether the
   policy already settles the question before collecting anything.
3. **Ambiguous source policy.** The model claimed weekend delivery was fine. The
   knowledge block only said "no *dispatch* on Sundays" — the model's reading was
   defensible. Fixed in the source text, not by scolding the model.

**Remaining known weakness:** the model still fabricates order state roughly
twice per ten conversations, and still paraphrases a policy number loosely
under pressure. The fabrication never reaches the customer, but it is being
*caught*, not *prevented* — the difference matters, and a larger model is the
real fix rather than more prompt text.

### 8.3 Failure handling

```bash
.venv/Scripts/python.exe scripts/failure_drill.py
```

[`scripts/failure_drill.py`](scripts/failure_drill.py) attacks the **live**
server with the real model behind it.

**Result: 15/15.** Full output: [`docs/failure-drill.txt`](docs/failure-drill.txt).

| Attack | Result |
|---|---|
| Unparseable JSON | `malformed_json`, socket stays open |
| JSON array instead of an object | `malformed_frame`, socket stays open |
| Unknown frame type | `invalid_frame`, socket stays open |
| `chat` frame with no `message` | `missing_message`, socket stays open |
| Whitespace-only message | `empty_message`, socket stays open |
| 5,000-character message | `invalid_frame`, socket stays open |
| `message` as a nested object, not a string | `invalid_frame`, socket stays open |
| Unknown session id over REST | `404` |
| Path traversal `/api/session/../../etc/passwd` | `404` |
| **Socket closed mid-stream after 3 tokens** | Server stays `ok`; the upstream generation is aborted by the generator teardown |
| **Second send on a session that is still generating** | `busy` — rejected immediately, not queued |
| **4 simultaneous sessions** | 4/4 completed in 38.3 s; `in_flight_generations` returns to 0 |

The design rule behind all of these: **no client input can close the
connection.** Every failure is a typed `error` frame on a still-usable socket,
so the UI can show a banner and the user can simply retry.



### 8.4 Concurrency under load

4 simultaneous WebSocket sessions, 3B model:

| Metric | Value |
|---|---|
| Sessions completed | **4 / 4** |
| Wall-clock for the batch | 54.6 s |
| TTFT mean / p95 | 28.1 s / 44.7 s |
| Total response mean / p95 | 34.9 s / 52.1 s |

**Nothing failed, but it got slow, and the reason is worth being precise
about.** This is not the API layer blocking — a REST call is served normally
while generations stream (asserted in
`test_one_slow_generation_does_not_block_other_requests`). Two things compound:

1. **One CPU, one model.** Six cores decoding four conversations share the same
   cores; per-user throughput drops roughly proportionally.
2. **KV cache thrashing — the dominant effect.** Each session has a different
   prompt tail, so four sessions fight over the runtime's cache slots and evict
   each other. Every turn then pays the full ~27 s uncached prompt evaluation
   from §8.1 instead of ~130 ms. That is why concurrent TTFT (28 s) lands almost
   exactly on the uncached single-turn figure.

Raising `OLLAMA_NUM_PARALLEL` and Ollama's cache-slot count would ease this;
the real fix is more hardware. Single-user latency (§8.1) is the number that
represents normal use, and the concurrency cap exists so that four users each
get a slow-but-complete answer rather than eight users all timing out.

---

## 9. Testing

```bash
.venv/Scripts/python.exe -m pytest
```

```
195 passed in 2.42s
```


The suite runs against the `MockEngine` double, so it is fast and needs no model.
What it covers:

| File | Covers |
|---|---|
| `tests/test_domain_policy.py` | Guard (injection, off-domain, false positives), intent taxonomy, slot extraction, malformed-reference detection |
| `tests/test_memory.py` | Token estimation, window planning, the verbatim floor, oversized turns, session store TTL/LRU/reset |
| `tests/test_manager.py` | Validation, guard short-circuit, stage machine, fact pinning across eviction, prompt-prefix stability, rolling summary + watermark, topic switch/resume, model failures, output scrubbing |
| `tests/test_api.py` | REST surface, WebSocket protocol, every error code, disconnect mid-stream, busy rejection, session isolation, non-blocking behaviour |
| `tests/test_fabrication_detector.py` | The output verifier across the whole verified/unverified matrix, in both directions |

Correctness against the **real** model is a separate script, because a mock
cannot tell you whether the assistant stays in character:

```bash
.venv/Scripts/python.exe scripts/evaluate.py -v
```

---

## 9b. Bonus claim — UX / persona polish

Claiming the **UX/persona polish** bonus (one only, per the brief), on two grounds:

**Why not the cloud-deployment option.** The brief allows one bonus, and the
alternative was deploying to a free-tier host such as Vercel. That is not
achievable for this system rather than merely inconvenient: Vercel runs
short-lived serverless functions with no persistent process, a bundle limit far
below the 1.9 GB of model weights, an execution ceiling in the tens of seconds
against generations that take ~25 s, and no support for the long-lived
WebSocket this API is built on. Any "deployment" would have had to call a hosted
model API, which the assignment forbids outright ("no cloud model APIs"). A
GPU-backed VM would work, but that is not free-tier and is not what the bonus
described. So the honest choice was the option this system can actually satisfy.

**Persona holds under adversarial testing.** `scripts/evaluate.py` includes four
adversarial scenarios — persona override, system-prompt extraction, pressure for
a policy exception, and a demand for a live order lookup. The persona survives
all of them, and the one failure mode that prompting could not eliminate
(inventing order status) is caught by a dedicated verification layer (§3.5,
layer 4) that replaces the reply before the customer sees it. The evaluation
reports how often that fires rather than hiding it inside the score.

**UI beyond what Phase V asks for.** Phase V requires streaming, history, reset
and a non-confusing layout. Beyond that: a **Telemetry** toggle exposing live
per-turn TTFT, tok/s, token counts and the context-window state
(`ctx 4/17 turns`, `evicted 13`, `pinned: email, order_id`) so the memory scheme
in §4.2 is visible while you use it; guarded and corrected replies tinted and
labelled so the safety mechanisms are legible rather than silent; an honest
empty state that tells the user up front there is no live order access;
auto-reconnect with backoff and a 25 s heartbeat; a stop control; session
resume across refresh; light/dark; responsive to ~375 px; and
`prefers-reduced-motion` respected. No framework and no build step — three files.

---

## 10. Known limitations

**Model quality.** This is a 3B model at 4-bit. It occasionally paraphrases a
policy number loosely, and its prose is more repetitive than a frontier model's.
The evaluation score in §8.2 is the honest measure — it is not 100%, and the
remaining failures are documented rather than hidden.

**No live order data — by design and by constraint.** Ava cannot look up an
order, because tools and RAG are out of scope for this assignment. She collects
and validates identifiers and applies policy. Assignment 2 is the natural place
to add retrieval.

**The guard is regex-based.** It is fast, free and consistent, but it is
pattern-matching: a sufficiently novel phrasing of a jailbreak gets through to
layer 2 (the system prompt), which is weaker. It also cannot catch off-domain
requests phrased in Nimbus vocabulary.

**Sessions are in-process and in-memory.** A restart loses every conversation,
and the server cannot be scaled to multiple workers without a shared store.
Bounded at 500 sessions with 1-hour TTL and LRU eviction, so it degrades
predictably rather than exhausting memory.

**Latency in real conversations is high.** ~8 s to first token and ~25 s to a
complete answer (§8.1) is usable for a demo and too slow for production. It is
not an architectural problem — it is ~64 tok/s of prompt evaluation on six CPU
cores against a ~2,500-token prompt. The fixes, in order of expected value:
extend the stable prompt prefix by reordering the volatile blocks, shrink the
policy block, or run on a GPU (the same model reached 188 tok/s on the RTX 3080
before CPU was enforced).

**Concurrency is bounded by one CPU and one model.** The API multiplexes cleanly
and never blocks, but Ollama decodes essentially serially on CPU. Four
simultaneous users each get roughly a quarter of the throughput — §8.3 measures
exactly this. It is a hardware limit, not an architectural one.

**Token counting is an estimate.** ~±10% (measured, §8.1). Budgets are set with
enough headroom to absorb it, and the estimator biases towards over-counting so
errors shorten the prompt rather than overflow the window.

**Rolling summaries are lossy.** Compressing to ~70 words drops nuance. Pinned
facts exist precisely because the identifiers that matter must not depend on
summary quality.

**One order per session.** Pinned facts are a flat dictionary, so a second order
reference given later in the same conversation overwrites the first. A customer
juggling two orders at once would confuse it. Parking a *lane* on a topic switch
is supported; parking a whole per-order fact set is not, and half-implementing
it would be worse than the current honest limit. Multi-order state is the
natural companion to the retrieval layer in Assignment 2.

**No authentication or rate limiting.** It binds to `127.0.0.1` and is a local
demo. Message size, session count and concurrency are capped, but there is no
per-IP throttling; this is not internet-facing software.

**Single language in practice.** The prompt tells Ava to answer in the
customer's language, but the policy block is English and a 3B model's Urdu
output quality is not something this project measured.

---

## 10b. Viva prep

**Where everything lives**

| File | Does | Likely question |
|---|---|---|
| [`domain/knowledge.py`](app/domain/knowledge.py) | Static Nimbus policy text | "Is this RAG?" No — no retrieval step, no index. It is a constant in the prompt. |
| [`domain/policy.py`](app/domain/policy.py) | Stages, lanes, guard, intent/slot regexes | "How do you refuse off-topic?" Four layers; see §3.5. |
| [`domain/prompts.py`](app/domain/prompts.py) | Builds the prompt | "Why is the system prompt fixed?" KV-cache prefix; see §8.1. |
| [`domain/verification.py`](app/domain/verification.py) | Blocks invented order data | "Why not just prompt it?" We tried three times; it never hit zero. |
| [`conversation/manager.py`](app/conversation/manager.py) | Orchestrates one turn | The main walkthrough file. |
| [`conversation/memory.py`](app/conversation/memory.py) | Picks which turns to replay | "What happens at turn 50?" Oldest evicted, facts pinned, summary rolls. |
| [`conversation/session.py`](app/conversation/session.py) | Session + bounded store | "Why bounded?" Client-supplied ids would otherwise exhaust memory. |
| [`llm/base.py`](app/llm/base.py) | The engine contract | "Why an interface?" Lets 166 tests run with no model in 2.5 s. |
| [`api/main.py`](app/api/main.py) | REST + WebSocket | "How do you not block?" Async throughout; CPU work is in Ollama's process. |

**Five decisions, and the reason for each**

1. **Fixed system prompt, turn context on the last message.** Turn-specific text in
   the system prompt put the history past the cache boundary and re-read the whole
   conversation each turn. Fixing it halved TTFT (§8.1).
2. **Facts pinned outside the history window.** A sliding window alone loses the
   order reference by turn nine. Pinning costs ~20 tokens and fixes it (§4.2).
3. **Guard detects deterministically, model writes the refusal.** Regex is
   consistent where a 3B model is not; keeping the wording model-generated keeps
   us inside "prompt orchestration alone" (§3.5).
4. **Output verification.** Prompting reduced invented order status but never
   removed it, so a narrow detector replaces those replies (§3.5, layer 4).
5. **3B over 1.5B.** Measured, not assumed: 92.5% vs 88% on the same suite, for
   ~1.8x slower decode (§4.1).

**Numbers to know:** ~8 s TTFT, ~12 tok/s decode, ~64 tok/s prompt eval, 4,096
context, ~1,780-token cached prefix, ~206 tokens re-evaluated per turn, 166 tests.

---

## 11. Submission

**Group of two**

| Roll number | Name |
|---|---|
| 22i-2530 | Ahmed Javaid |
| 22i-2470 | Ammad Ashraf |

**Contents of this repository**

| Item | Where |
|---|---|
| Backend source | [`app/`](app/) — API, conversation manager, domain layer, LLM engines |
| Frontend source | [`frontend/`](frontend/) — `index.html`, `app.js`, `styles.css` |
| Test cases | [`tests/`](tests/) — 166 tests, `python -m pytest` |
| Scripts | [`scripts/`](scripts/) — benchmarks, evaluation, failure drill, transcripts |
| Documentation | this README, plus [`docs/`](docs/) for raw benchmark and evaluation output |

**Reproducing every number in this README**

```bash
.venv/Scripts/python.exe -m pytest
```

```bash
.venv/Scripts/python.exe scripts/benchmark.py --all --rounds 3 --concurrent 4
```

```bash
.venv/Scripts/python.exe scripts/evaluate.py -v
```

```bash
.venv/Scripts/python.exe scripts/failure_drill.py
```

The last two need Ollama running with the model pulled; the test suite does not.

**Declaration.** All work in this repository is our own. Generative AI was used
as a coding assistant, as the brief permits; we understand the code and can
explain and reproduce it.

---

## 12. Repository layout

```
assignment-1/
├── app/
│   ├── config.py                 # all settings, env-driven
│   ├── api/
│   │   ├── main.py               # FastAPI app: REST + /ws/chat + static UI
│   │   └── schemas.py            # pydantic wire format, both directions
│   ├── conversation/
│   │   ├── manager.py            # turn orchestration, stage machine, scrubbing
│   │   ├── memory.py             # context-window planning
│   │   ├── session.py            # Session, Turn, bounded SessionStore
│   │   └── tokens.py             # token estimation
│   ├── domain/                   # everything Nimbus-specific lives here
│   │   ├── knowledge.py          # static policy/catalogue block (no retrieval)
│   │   ├── policy.py             # stages, lanes, guard, intent + slot extraction
│   │   ├── prompts.py            # structured system-prompt construction
│   │   └── verification.py       # blocks invented order data reaching the user
│   └── llm/
│       ├── base.py               # the engine contract
│       ├── ollama_engine.py      # primary backend, async streaming
│       ├── llamacpp_engine.py    # optional alternative backend
│       ├── mock_engine.py        # deterministic test double
│       └── factory.py            # engine selection
├── frontend/                     # index.html · app.js · styles.css
├── scripts/
│   ├── benchmark.py              # latency, context scaling, concurrency, calibration
│   ├── evaluate.py               # correctness against the real model
│   ├── make_screenshots.py       # captures the README images
│   ├── failure_drill.py          # breaks the live server on purpose
│   └── make_transcripts.py       # captures the README's example dialogues
├── tests/                        # pytest suite (mock engine)
├── docs/
│   ├── example-dialogues.md      # real captured transcripts
│   ├── benchmarks-3b.md          # raw benchmark output, selected model
│   ├── benchmarks-1.5b.md        # raw benchmark output, the alternative
│   ├── benchmark-conversation.md # per-turn latency, real conversation
│   ├── failure-drill.txt         # failure-handling drill output
│   └── screenshots/              # README images, captured from the live app
├── requirements.txt              # pinned
├── .env.example
└── README.md
```
