# Price Intelligence Agent

Compares a product across marketplaces worldwide, normalises the results into one
currency, and **waits for a human to approve the result before dispatching it**.

Built as a LangGraph state machine so the approval gate is part of the graph
rather than a `input()` buried in the middle of a function: the run suspends at a
real interrupt, its state is checkpointed, and it resumes from the same place
when a decision arrives — even across process restarts.

---

## The pipeline

```
START ─▶ extract ─▶ synthesize ─▶ guardrail ─┬─▶ dispatch ─▶ END
                                 (interrupt)  └─▶ END (rejected)
```

| Node | Does |
|------|------|
| **extract** | Tavily search across the marketplaces for the user's region, then LLM structured extraction into a Pydantic schema. |
| **synthesize** | Converts every listing into the user's home currency, sorts, and builds a `ComparisonReport`. |
| **guardrail** | `interrupt()` — suspends for human review. Approve or reject comes back as the decision. |
| **dispatch** | POSTs the approved payload to a webhook. A bundled local receiver stands in when no `WEBHOOK_URL` is set. |

`extract` and `synthesize` are injectable, which is what makes `--fixture` work.

---

## Design decisions worth knowing

**Currency is not converted with a hardcoded table alone.** `LIVE_FX=1` (default)
pulls live rates from `open.er-api.com` and falls back to the fixed table in
`schemas.py` on any failure, so the feature degrades instead of breaking.

**A listing has two prices.** `price_local` is what the marketplace charges;
`landed_local` adds cross-border shipping estimates. Sorting by `price` and
sorting by `landed` routinely disagree — AliExpress often wins on sticker price
and loses on landed. Both are available as sort keys.

**The LLM never sees a price it invents.** Extraction is constrained to a Pydantic
model, and `_invoke_with_backoff` retries only transient failures (429/503/404)
rather than blindly retrying a malformed response.

---

## Setup

```bash
git clone <this-repo> && cd Price
uv venv && uv pip install -e .

cp .env.example .env      # then fill in GEMINI_API and TAVILY_API_KEY
```

`LLM_PROVIDER` selects the extractor: `gemini` (default) or `openai`.

---

## Run it

**No keys? Use fixture mode** — offline sample data, no network, no API keys:

```bash
uv run python cli.py "Samsung EB-BG965ABE battery" --fixture
```

**Live search:**

```bash
uv run python cli.py "Samsung EB-BG965ABE battery" --region PK
uv run python cli.py "RTX 4090" --region AE --sort landed --target AED
uv run python cli.py "iPhone 16 Pro" --json > report.json
```

`--auto-approve` / `--auto-reject` resume the interrupt without a human, for
scripted or CI use.

**Streamlit UI:**

```bash
uv run streamlit run app.py
```

The UI streams node progress, shows the comparison table, and gives you the
approve/reject prompt inline.

---

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--region` | `PK` | Preset marketplace mix + home currency |
| `--marketplace` | — | Explicit IDs, repeatable; overrides `--region` |
| `--sort` | `price` | `price`, `landed`, `sales`, `stars`, `reviews` |
| `--target` | region currency | Convert everything to this instead |
| `--fixture` | off | Offline sample data |

Regions: `PK` `DE` `US` `UK` `CN` `NG` `KE` `AE` `SA` `IN`

---

## Webhook

Point `WEBHOOK_URL` at Slack, Teams, Pipedream, `webhook.site`, or anything that
accepts a POST. With it unset, `webhook_server.py` starts a local receiver on
`WEBHOOK_PORT` (default `8504`) and logs to `webhook_events.json`, so
dispatch is still exercised end to end.

Only approved reports are dispatched — a rejection ends the run at the guardrail.

---

## Environment

| Variable | Required | Notes |
|----------|----------|-------|
| `GEMINI_API` | if `LLM_PROVIDER=gemini` | Primary extractor |
| `OPENAI_API_KEY` | if `LLM_PROVIDER=openai` | Alternative extractor |
| `TAVILY_API_KEY` | for live search | Legacy `TAVILY_API` also honoured |
| `TAVILY_DEPTH` | no | `advanced` (default) for fuller snippets |
| `WEBHOOK_URL` | no | Empty = bundled local receiver |
| `LIVE_FX` | no | `1` live rates (default), `0` fixed table |

`.env` is gitignored. `.env.example` is the committed template.

---

## Layout

| File | Role |
|------|------|
| `agent_graph.py` | LangGraph nodes, edges, provider fallback, checkpointer |
| `schemas.py` | Pydantic models, FX, sorters |
| `tools.py` | Tavily search + FX fetch |
| `marketplaces.py` | Marketplace registry, region presets |
| `cli.py` | Headless runner |
| `app.py` | Streamlit UI |
| `webhook_server.py` | Local dispatch receiver |
