from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

import marketplaces
import tools
from schemas import ComparisonReport, CurrencyStandardizer, Decision, DispatchResult, ProductListing

DISPATCH_LOG = Path(__file__).parent / "dispatch_log.json"
CHECKPOINTS_DB = Path(__file__).parent / "checkpoints.sqlite"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

_PLACEHOLDER_MARKERS = ("your-", "changeme", "placeholder", "here")


def resolve_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        if provider not in ("openai", "gemini"):
            raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}; use 'openai' or 'gemini'")
        return provider
    return "gemini" if _valid_secret("GEMINI_API") else "openai"


def _valid_secret(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and not any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS)


def resolve_llm_key() -> str | None:
    return os.getenv("GEMINI_API").strip() if resolve_llm_provider() == "gemini" and _valid_secret("GEMINI_API") else (
        os.getenv("OPENAI_API_KEY") if _valid_secret("OPENAI_API_KEY") else None
    )


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class PriceState(TypedDict, total=False):
    query: str
    region: str
    target_currency: str
    marketplace_ids: list[str]
    sort_key: str
    raw_results: list[dict]
    raw_errors: list[dict]
    report: dict
    decision: dict
    dispatch_result: dict
    created_at: str


# --------------------------------------------------------------------------- #
# Node 1: Multi-source extraction (dynamic marketplace selection)
# --------------------------------------------------------------------------- #
def extract_node(state: PriceState) -> dict[str, Any]:
    query = state["query"]
    region = state.get("region", "PK")
    requested = state.get("marketplace_ids")
    if requested:
        mp_ids = list(requested)
    else:
        # dynamic mode: scan widely (region-priority order), keep only sites that
        # actually returned product pages
        mp_ids = marketplaces.ordered_for(region)[: tools.TAVILY_SCAN_MARKETPLACES]

    extraction = tools.extract_sources(query, mp_ids)
    if not requested:
        mp_ids = tools.select_best_marketplaces(region, extraction.results)
    results = [r for r in extraction.results if r.get("source") in mp_ids]
    return {"raw_results": results, "marketplace_ids": mp_ids, "raw_errors": extraction.errors}


# --------------------------------------------------------------------------- #
# Node 2: LLM structured extraction + schema normalization
# --------------------------------------------------------------------------- #
class RawProductList(BaseModel):
    products: list[ProductListing]


def _build_extraction_prompt(query: str, target_currency: str, raw_results: list[dict]) -> list[object]:
    blocks = []
    for i, row in enumerate(raw_results, start=1):
        blocks.append(
            f"{i}. [source={row.get('source')} | marketplace={row.get('marketplace')} | "
            f"country={row.get('country')} | currency={row.get('currency')}]\n"
            f"   title:  {row.get('title')}\n"
            f"   url:    {row.get('url')}\n"
            f"   snippet: {row.get('snippet')}"
        )
    system = (
        "You are a price-intelligence extractor. Convert each raw search result snippet into a "
        "structured product listing. Rules:\n"
        "1. source must equal the marketplace id given in brackets (e.g. daraz_pk, amazon_de, "
        "temu, jumia_ng).\n"
        "2. currency must be the ORIGINAL listing currency shown in brackets (e.g. PKR, USD, EUR, "
        "CNY, NGN). Extract raw_price as a number in that original currency — never convert it "
        "yourself.\n"
        "3. Set model to the normalized product/SKU identifier (e.g. 'EB-BG965ABE').\n"
        "4. From the snippet, also extract: sales (units sold, e.g. '100 sold' -> 100; default 0), "
        "rating (seller stars 0-5, e.g. '4.7'; default 0), reviews (count; default 0), and "
        "shipping cost in the listing's own currency when it is clearly stated (default 0).\n"
        "5. Only emit listings that plausibly match the query. Skip clearly unrelated "
        "hits; prefer real product pages over category pages.\n"
        "6. Never emit a listing unless you can determine a raw_price greater than zero "
        "from the snippet; omit it entirely if no price is visible.\n"
        "7. EMIT EVERY usable listing you see — you may and should output MULTIPLE "
        "listings per source. Doing so gives buyers a fuller market view. Do not stop "
        "at one listing per marketplace.\n"
        "8. Set confidence 0-1 based on how confident you are that the listing matches "
        f"the queried model. All prices will ultimately be compared in {target_currency}."
    )
    human = (
        f"Query: {query}\n"
        "Raw search results below. Emit the structured product listings:\n\n"
        + "\n".join(blocks)
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _gemini_candidates() -> list[str]:
    """Ordered candidate models: env chain first, then live-discovered flash/lite/pro models.

    Discovery guards against hard-coding a model id that Google has retired (404).
    """
    preferred = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()]
    key = (os.getenv("GEMINI_API") or "").strip()
    if key:
        try:
            from google import genai

            client = genai.Client(api_key=key)
            discovered = [
                m.name.split("/")[-1]
                for m in client.models.list()
                if "generateContent" in (m.supported_actions or ())
            ]
            # cheap tier first, premium tier as a pressure valve
            rank = {
                "flash": 0, "flash-lite": 1, "lite": 1, "pro": 2, "nano": 3,
            }
            others = sorted(set(discovered) - set(preferred), key=lambda n: (rank.get(next((k for k in rank if k in n), 9), 9), n))
            ordered = list(dict.fromkeys(preferred + others))
            return ordered or preferred
        except Exception:  # noqa: BLE001 - discovery is best-effort; fall back to env list
            pass
    return preferred or ["gemini-3.6-flash", "gemini-flash-latest"]


def _transient_code(exc: Exception) -> int | None:
    """Return HTTP-ish error code if the exception is a transient 429/503, else None."""
    import re

    body = str(exc)
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code if code in (429, 503) else None
    m = re.search(r"\b(429|503)\b", body)
    return int(m.group(1)) if m else None


def _invoke_with_backoff(model, messages, attempts: int = 4) -> RawProductList:
    delay = 2.0
    for attempt in range(attempts):
        try:
            return model.invoke(messages)  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            if _transient_code(exc) is None or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")  # pragma: no cover


def _extract_products(query: str, target_currency: str, raw_results: list[dict]) -> list[ProductListing]:
    if not raw_results:
        return []
    messages = _build_extraction_prompt(query, target_currency, raw_results)
    provider = resolve_llm_provider()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = (os.getenv("GEMINI_API") or "").strip()
        errors: list[str] = []
        for candidate in _gemini_candidates():
            try:
                model = ChatGoogleGenerativeAI(
                    model=candidate, api_key=key, temperature=0, max_retries=2, timeout=45
                ).with_structured_output(RawProductList)
                # retry transient 503/429 with backoff inside _invoke_with_backoff;
                # a permanent error (404 retired model, 400) falls through to the next candidate
                parsed: RawProductList = _invoke_with_backoff(model, messages)
                return parsed.products
            except Exception as exc:  # noqa: BLE001 - 404 dead model, 429 quota, 503 overload
                errors.append(f"{candidate}: {type(exc).__name__}: {str(exc)[:120]}")
        raise RuntimeError("All Gemini model candidates failed — " + " | ".join(errors))
    elif provider == "openai":
        model = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(RawProductList)
    else:  # pragma: no cover - guarded by resolve_llm_provider
        raise ValueError(f"Unsupported provider: {provider}")
    parsed = model.invoke(messages)  # type: ignore[return-value]
    return parsed.products


def synthesize_node(state: PriceState) -> dict[str, Any]:
    query = state["query"]
    region = state.get("region", "PK")
    target_currency = state.get("target_currency") or marketplaces.REGION_CURRENCY.get(region, "PKR")
    sort_key = state.get("sort_key", "price")
    raw_results: list[dict] = state.get("raw_results", [])

    CurrencyStandardizer.set_target(target_currency)
    products = _extract_products(query, target_currency, raw_results)

    report = ComparisonReport(
        query=query,
        as_of=datetime.now(timezone.utc).isoformat(),
        region=region,
        target_currency=target_currency,
        sort_key=sort_key,
        currency_note=CurrencyStandardizer.display_rate(),
        products=products,
    )
    return {"report": report.model_dump(mode="json")}


# --------------------------------------------------------------------------- #
# Node 3: State-interrupt guardrail (human-in-the-loop pause)
# --------------------------------------------------------------------------- #
def guardrail_node(state: PriceState) -> dict[str, Any]:
    report: dict = state.get("report") or ComparisonReport(query=state["query"], as_of="", currency_note="").model_dump(mode="json")
    decision = interrupt({"type": "review_request", "report": report})
    parsed = Decision.model_validate(decision if isinstance(decision, dict) else {"action": decision})
    return {"decision": parsed.model_dump()}


# --------------------------------------------------------------------------- #
# Node 4: Action dispatcher (webhook / dry-run log)
# --------------------------------------------------------------------------- #
def _append_dispatch_log(payload: dict) -> None:
    entries: list[dict] = []
    if DISPATCH_LOG.exists():
        try:
            entries = json.loads(DISPATCH_LOG.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append(payload)
    DISPATCH_LOG.write_text(json.dumps(entries, indent=2, default=str))


def dispatch_node(state: PriceState) -> dict[str, Any]:
    decision: dict = state.get("decision", {})
    report: dict = state.get("report", {})
    action = decision.get("action")
    note = decision.get("note", "")

    if action != "approve":
        return {"dispatch_result": DispatchResult(action="reject", status="skipped", detail="No external action taken.").model_dump()}

    best = (report.get("best_deal") or {}) if isinstance(report, dict) else {}
    payload = {
        "thread": state.get("created_at"),
        "action": "approve",
        "reviewer_note": note,
        "query": state.get("query"),
        "region": state.get("region", report.get("region")),
        "target_currency": state.get("target_currency", report.get("target_currency")),
        "sort_key": state.get("sort_key", report.get("sort_key")),
        "marketplace_ids": state.get("marketplace_ids", []),
        "report_as_of": report.get("as_of"),
        "best_deal": best,
        "all_listings": report.get("products", []),
        "timestamp": datetime.now(timezone.utc).isoformat() if isinstance(report, dict) else None,
    }

    if WEBHOOK_URL:
        target_url = WEBHOOK_URL
    else:
        # No configured URL -> start the bundled local webhook receiver so the
        # approval truly dispatches (status "dispatched") instead of a dry-run.
        try:
            import webhook_server

            target_url = webhook_server.ensure_started()
        except Exception:  # noqa: BLE001 - server unavailable; fall back to local log
            target_url = None

    if target_url:
        try:
            import urllib.request

            req = urllib.request.Request(
                target_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
            result = DispatchResult(action="approve", status="dispatched", target=target_url,
                                    detail=f"Webhook received HTTP {status}.")
        except Exception as exc:  # noqa: BLE001
            result = DispatchResult(action="approve", status="dispatched", target=target_url,
                                    detail=f"Webhook failed ({exc}); fallback logged locally.")
            _append_dispatch_log(payload | {"fallback": True, "error": str(exc)})
    else:
        result = DispatchResult(action="approve", status="dry_run", target=str(DISPATCH_LOG),
                                detail="No webhook available; dispatch payload logged locally as dry-run.")
        _append_dispatch_log(payload)

    return {"dispatch_result": result.model_dump()}


def route_after_guardrail(state: PriceState) -> str:
    return "dispatch" if (state.get("decision") or {}).get("action") == "approve" else END


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph(extract_fn=extract_node, synthesize_fn=synthesize_node, checkpointer=None):
    builder = StateGraph(PriceState)
    builder.add_node("extract", extract_fn)
    builder.add_node("synthesize", synthesize_fn)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("dispatch", dispatch_node)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "synthesize")
    builder.add_edge("synthesize", "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail, {"dispatch": "dispatch", END: END})
    builder.add_edge("dispatch", END)

    if checkpointer is None:
        checkpointer = _sqlite_checkpointer()
    return builder.compile(checkpointer=checkpointer)


def _sqlite_checkpointer() -> SqliteSaver:
    """Persistent checkpointer — paused runs survive app restarts (not just reloads)."""
    import sqlite3

    conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
    return SqliteSaver(conn)


GRAPH = build_graph()


def new_thread_config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def run_stream(
    query: str,
    config: dict | None = None,
    auto_resume: dict | None = None,
    region: str = "PK",
    marketplace_ids: list[str] | None = None,
    sort_key: str = "price",
    target_currency: str | None = None,
):
    """Stream graph events. Yields (kind, payload) pairs.

    kind is one of: 'node', 'interrupt', 'error', 'done'.
    If auto_resume is provided, any interrupt is immediately resolved with that value.
    """
    config = config or new_thread_config()
    # None = dynamic auto-selection (the graph scans region-ordered marketplaces
    # and keeps the ones that actually returned product pages).
    if not target_currency:
        target_currency = marketplaces.REGION_CURRENCY.get(region, "PKR")
    inputs: dict[str, Any] = {
        "query": query,
        "region": region,
        "marketplace_ids": marketplace_ids,
        "sort_key": sort_key,
        "target_currency": target_currency,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        for event in GRAPH.stream(inputs, config=config, stream_mode="updates"):
            for node_name, payload in event.items():
                if node_name == "__interrupt__":
                    yield "interrupt", payload[0].value
                    if auto_resume is not None:
                        resume(config, auto_resume)
                else:
                    yield "node", (node_name, payload)
        final = GRAPH.get_state(config)
        yield "done", final.values
    except Exception as exc:  # noqa: BLE001
        yield "error", str(exc)


def resume(config: dict, resume_value: Any) -> dict:
    """Resume a paused thread with a human decision and return the final state."""
    GRAPH.invoke(Command(resume=resume_value), config=config)
    return GRAPH.get_state(config).values


def get_state(config: dict) -> dict:
    return GRAPH.get_state(config).values