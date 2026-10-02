from __future__ import annotations

import html
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

import agent_graph
import marketplaces
import tools
import webhook_server

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

WEBHOOK_EVENTS = Path(__file__).parent / "webhook_events.json"


# --------------------------------------------------------------------------- #
# Brand styling — custom components use their own self-contained dark "glass"
# panels with light text baked in, so they read correctly on ANY Streamlit theme
# (no theme detection, no white-on-white).
# --------------------------------------------------------------------------- #
def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, .stApp * { font-family: 'Inter', -apple-system, system-ui, sans-serif; }

.hero { background: linear-gradient(135deg, #0f172a 0%, #14532d 130%);
        border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 30px 32px 26px;
        margin-bottom: 20px; color: #f8fafc; }
.hero h1 { font-family: 'Sora', sans-serif; font-weight: 800; letter-spacing: -0.5px;
           font-size: 2.1rem; margin: 0 0 6px; color: #f8fafc; }
.hero .sub { color: #cbd5e1; font-size: 1rem; margin-bottom: 14px; }
.hero .badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge { display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: .76rem;
         font-weight: 600; border: 1px solid rgba(255,255,255,.18); color: #e2e8f0;
         background: rgba(15,23,42,.45); }
.badge.live { background: rgba(16,185,129,.16); border-color: #10b981; color: #34d399; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid rgba(128,128,128,.25); }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: #10b981; }

.pgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px; }
.pcard { position: relative; display: block; text-decoration: none !important; color: #e8eef7 !important;
         background: #1c2330; border: 1px solid rgba(255,255,255,.10); border-radius: 16px;
         padding: 18px 18px 14px;
         transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; }
.pcard:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(0,0,0,.35); border-color: #10b981; }
.pcard.best { background: #12251b; border-color: #10b981; }
.ribbon { position: absolute; top: -1px; right: -1px; background: #10b981; color: #ffffff;
          font-size: .68rem; font-weight: 700; padding: 5px 10px; border-radius: 0 15px 0 12px; letter-spacing: .4px; }
.pc-head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.rank { flex: 0 0 auto; width: 26px; height: 26px; border-radius: 7px; display: inline-flex;
        align-items: center; justify-content: center; font-weight: 700; font-size: .82rem;
        color: #93c5fd; background: rgba(52,211,153,.14); }
.chip { font-size: .74rem; font-weight: 600; color: #cbd5e1;
        border: 1px solid rgba(255,255,255,.14); border-radius: 999px; padding: 3px 9px; }
.pc-title { font-size: .95rem; line-height: 1.35; font-weight: 600; min-height: 40px; color: #e8eef7; }
.pc-price { margin: 12px 0 8px; }
.amt { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 800; color: #34d399; }
.landed { display: block; font-size: .78rem; color: #94a3b8; margin-top: 2px; }
.pc-meta { font-size: .78rem; color: #94a3b8; display: flex; flex-wrap: wrap; gap: 4px 12px; }
.pc-action { margin-top: 12px; padding: 9px 14px; border-radius: 10px; text-align: center;
             font-weight: 600; font-size: .84rem; color: #ffffff;
             background: linear-gradient(135deg, #10b981, #4f46e5); transition: opacity .12s ease; }
.pcard:hover .pc-action { opacity: .88; }

.srcchip { display: inline-grid; grid-template-columns: auto auto auto; gap: 4px 14px; padding: 10px 16px;
           border: 1px solid rgba(255,255,255,.12); border-radius: 12px; background: #1c2330;
           font-size: .86rem; color: #e8eef7; }
a { color: #10b981; }
</style>
""",
        unsafe_allow_html=True,
    )


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _hero(region: str, target: str) -> None:
    st.markdown(
        f"""
<div class="hero">
  <h1>Price Intelligence Agent</h1>
  <div class="sub">Cross-marketplace price research with human-in-the-loop dispatch — compare local &amp;
     cross-border sources and land on one normalized number in your own currency.</div>
  <div class="badges">
    <span class="badge live">● LIVE</span>
    <span class="badge">Region · {marketplaces.REGION_NAMES.get(region, region)}</span>
    <span class="badge">Compare in {target}</span>
    <span class="badge">FX · {st.session_state.get("fx_note") or "—"}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Background worker: single owner of graph execution + resume (no concurrent
# invocations on the same thread). Blocked at the interrupt gate until the UI
# submits an approve/reject decision.
# --------------------------------------------------------------------------- #
class ReviewGate:
    def __init__(self) -> None:
        self._holder: dict | None = None
        self._event = threading.Event()

    def submit(self, value: dict) -> None:
        self._holder = value
        self._event.set()

    def wait(self) -> dict:
        self._event.wait(timeout=3600 * 24)  # wait indefinitely for human review
        return self._holder

    def clear(self) -> None:
        self._holder = None
        self._event.clear()


def _worker(query: str, config: dict, q: "queue.Queue[tuple[str, Any]]", gate: ReviewGate, run_params: dict) -> None:
    try:
        for kind, payload in agent_graph.run_stream(query, config, **run_params):
            if kind == "interrupt":
                q.put(("interrupt", payload))
                decision = gate.wait()
                agent_graph.resume(config, decision)
                gate.clear()
            else:
                q.put((kind, payload))
    except Exception as exc:  # noqa: BLE001
        log.exception("worker failed")
        q.put(("error", str(exc)))


def _drain(q: "queue.Queue[tuple[str, Any]]") -> list[tuple[str, Any]]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            return events


def _start_run(query: str, run_params: dict) -> None:
    config = agent_graph.new_thread_config()
    events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    gate = ReviewGate()
    threading.Thread(target=_worker, args=(query, config, events, gate, run_params), daemon=True).start()

    st.session_state["phase"] = "running"
    st.session_state["query"] = query
    st.session_state["run_params"] = run_params
    st.session_state["config"] = config
    st.session_state["events"] = events
    st.session_state["gate"] = gate
    st.session_state["nodes"] = []
    st.session_state["report"] = None
    st.session_state["interrupt_payload"] = None
    st.session_state["dispatch_result"] = None
    st.session_state["run_error"] = None
    st.session_state["raw_results"] = []
    st.session_state["raw_errors"] = []
    st.session_state["selected_mps"] = []


def _render_status(nodes: list[str]) -> None:
    labels = {
        "extract": "Node 1 · Scan marketplaces",
        "synthesize": "Node 2 · Extract & normalize",
        "guardrail": "Node 3 · Review gate",
        "dispatch": "Node 4 · Dispatch",
    }
    for node in ["extract", "synthesize", "guardrail", "dispatch"]:
        state = "✓" if node in nodes else ("⟳" if nodes and nodes[-1] == node else "○")
        st.markdown(f"- **{state}** {labels[node]}")


def _format_num(v: Any) -> str:
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- #
# Rendering — clickable product cards, comparison matrix, raw sources
# --------------------------------------------------------------------------- #
def _card(p: dict, rank: int, target: str, best: bool) -> str:
    url = _esc(p.get("url", ""))
    title = _esc(p.get("title") or p.get("model") or "Untitled listing")
    mp = _esc(p.get("marketplace") or p.get("source") or "—")
    country = _esc(p.get("country") or "")
    price = float(p.get("price_local") or 0) or float(p.get("raw_price") or 0)
    landed = float(p.get("landed_local") or 0) or price
    shipping = p.get("shipping")
    sales = _format_num(p.get("sales"))
    rating = float(p.get("rating") or 0)
    reviews = _format_num(p.get("reviews"))
    conf = float(p.get("confidence") or 0)
    ship_txt = f"ship {_esc(p.get('currency') or target)} {_format_num(shipping)}" if shipping else "free shipping"
    stars = ("★" + str(rating).rstrip("0").rstrip(".")) if rating else "no rating"
    ribbon = '<span class="ribbon">BEST DEAL</span>' if best else ""
    return (
        f'<a class="pcard{" best" if best else ""}" href="{url}" target="_blank" rel="noopener noreferrer">'
        f"{ribbon}"
        f'<div class="pc-head"><span class="rank">{rank}</span>'
        f'<span class="chip">{mp} · {_esc(country)}</span>'
        f'<span style="margin-left:auto;font-size:.74rem;color:#94a3b8" title="match confidence">{conf * 100:.0f}%</span></div>'
        f'<div class="pc-title">{title}</div>'
        f'<div class="pc-price"><span class="amt">{_esc(target)} {price:,.2f}</span>'
        f'<span class="landed">landed cost {_esc(target)} {landed:,.2f} · {ship_txt}</span></div>'
        f'<div class="pc-meta"><span>{sales} sold</span><span>{stars}</span><span>{reviews} reviews</span></div>'
        f'<div class="pc-action">Open listing on {mp} ↗</div>'
        f"</a>"
    )


def _render_cards(report: dict, target: str) -> None:
    products = report.get("products", [])
    if not products:
        st.warning(report.get("summary", "No usable product matches were extracted."))
        return
    best = report.get("best_deal") or {}
    cards = []
    for rank, p in enumerate(products, start=1):
        is_best = (
            (p.get("model"), p.get("source"), p.get("title"))
            == (best.get("model"), best.get("source"), best.get("title"))
        )
        cards.append(_card(p, rank, target, is_best))
    st.markdown('<div class="pgrid">' + "\n".join(cards) + "</div>", unsafe_allow_html=True)


def _render_matrix(report: dict) -> None:
    products = report.get("products", [])
    if not products:
        st.warning(report.get("summary", "No usable product matches were extracted."))
        return
    target = report.get("target_currency", "PKR")
    df = pd.DataFrame(
        [
            {
                "Listing": p.get("title", ""),
                "Marketplace": p.get("marketplace", p.get("source")),
                "Country": p.get("country", ""),
                "Price": f'{float(p.get("price_local") or 0):,.2f}',
                "Landed": f'{float(p.get("landed_local") or float(p.get("price_local") or 0)):,.2f}',
                "Sales": p.get("sales"),
                "Stars": p.get("rating"),
                "Conf": f'{float(p.get("confidence") or 0):.0%}',
                "URL": p.get("url", ""),
            }
            for p in products
        ]
    )
    best = report.get("best_deal")
    best_key = None
    if best:
        for idx, p in enumerate(products):
            if (p.get("model"), p.get("source"), p.get("title")) == (best.get("model"), best.get("source"), best.get("title")):
                best_key = idx
                break
    dark = st.get_option("theme.base") == "dark"

    def _style(row):
        if row.name != best_key:
            return ["" for _ in row]
        css = (
            "background-color:#16301c;color:#d9f6e3;font-weight:700"
            if dark
            else "background-color:#eafaf0;color:#14532d;font-weight:700"
        )
        return [css for _ in row]

    st.caption(f"Ranked by `{report.get('sort_key', 'price')}` — prices in {target}")
    st.dataframe(
        df.style.apply(_style, axis=1),
        column_config={
            "Listing": st.column_config.TextColumn("Listing", width="large"),
            "URL": st.column_config.LinkColumn("Open", width="small", display_text="View ↗"),
        },
        width="stretch",
        hide_index=True,
    )


def _render_sources(report: dict) -> None:
    raw = st.session_state.get("raw_results", [])
    selected = st.session_state.get("selected_mps", [])
    errors = st.session_state.get("raw_errors", [])
    st.markdown("##### Marketplaces with product-page results")
    if selected:
        chips = "".join(
            f"<span><strong>{_esc(marketplaces.get(i).name)}</strong></span>"
            f"<span>{_esc(marketplaces.get(i).country)}</span>"
            f"<span>{_esc(marketplaces.get(i).currency)}</span>"
            for i in selected
        )
        st.markdown(f'<div class="srcchip">{chips}</div>', unsafe_allow_html=True)
    if errors:
        st.warning(f"{len(errors)} source(s) failed and were skipped.")
    if not raw:
        st.info("No raw snippets kept — nothing passed the product-page filter for this query.")
        return
    for row in raw:
        with st.expander(f"[{_esc(row.get('source'))}] {_esc(row.get('title'))[:80]}"):
            st.markdown(f"**URL:** {_esc(row.get('url'))}")
            st.markdown(f"**Snippet:** {_esc(row.get('snippet'))}")


# --------------------------------------------------------------------------- #
# Review / dispatch rendering
# --------------------------------------------------------------------------- #
def _render_review(report: dict) -> None:
    target = report.get("target_currency", "PKR")
    n = len(report.get("products", []))
    st.session_state["fx_note"] = report.get("currency_note", "")
    tab_cards, tab_matrix, tab_sources = st.tabs([f"Offers ({n})", "Comparison matrix", "Raw sources"])
    with tab_cards:
        _render_cards(report, target)
    with tab_matrix:
        _render_matrix(report)
    with tab_sources:
        _render_sources(report)
    st.caption(f"FX note: {report.get('currency_note', '')} · ranked by `{report.get('sort_key', 'price')}`")


def _render_dispatch(result: dict) -> None:
    st.subheader("Dispatch result")
    status = result.get("status")
    target = result.get("target", "—")
    detail = result.get("detail", "")
    if status == "dispatched":
        st.success(f"**Approved action dispatched** → `{target}`\n\n{detail}")
    elif status == "dry_run":
        st.warning(f"**Dry-run** — payload appended to `{target}`\n\n{detail}")
    else:
        st.info("**Rejected** — no external action was taken.")

    events = []
    if WEBHOOK_EVENTS.exists():
        try:
            events = json.loads(WEBHOOK_EVENTS.read_text())
        except (json.JSONDecodeError, OSError):
            events = []
    if events:
        with st.expander(f"Webhook feed — {len(events)} received event(s)"):
            st.json(events[-1])


# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Price Intelligence Agent", page_icon="🔎", layout="wide")
_inject_css()

default_query = "Samsung Galaxy S9+ Battery EB-BG965ABE"
region_codes = sorted(marketplaces.REGION_DEFAULTS)

with st.form("query_form"):
    col_query, col_region, col_sort = st.columns([3, 1, 1])
    query = col_query.text_input("Product name / SKU", value=default_query)
    region = col_region.selectbox(
        "Your region", options=region_codes, index=region_codes.index("PK"),
        format_func=lambda r: marketplaces.REGION_NAMES[r],
    )
    sort_key = col_sort.selectbox(
        "Rank by", options=["price", "landed", "sales", "stars", "reviews"],
        format_func=lambda s: {"price": "Cheapest first", "landed": "Cheapest incl. shipping",
                               "sales": "Most sales", "stars": "Highest rated", "reviews": "Most reviews"}[s],
    )
    auto = st.checkbox(
        "Auto-select best marketplaces",
        value=True,
        help="Scans every region-relevant marketplace (local first) and keeps only the sites "
             "that actually have product results for your query.",
    )
    mps = []
    if auto:
        st.caption("Dynamic discovery: scanning all region-relevant marketplaces — local shops and "
                   "cross-border platforms. The best sources are picked per query.")
    else:
        mps = st.multiselect(
            "Marketplaces",
            options=marketplaces.all_ids(),
            default=marketplaces.default_for(region),
            format_func=lambda i: f"{marketplaces.get(i).name} — {marketplaces.get(i).country} ({marketplaces.get(i).currency})",
            key=f"mps_{region}",
            help="Pick exactly which shops to compare.",
        )
    submitted = st.form_submit_button("Start analysis", type="primary")

_hero(region, marketplaces.REGION_CURRENCY[region])

if submitted and query.strip():
    missing = []
    provider = agent_graph.resolve_llm_provider()
    if not agent_graph.resolve_llm_key():
        missing.append(f"LLM key for provider '{provider}' (GEMINI_API / OPENAI_API_KEY)")
    if not tools.resolve_tavily_key():
        missing.append("TAVILY_API_KEY")
    if not auto and not mps:
        missing.append("at least one marketplace")
    if missing:
        st.error(f"Missing: {', '.join(missing)}. Fix in `.env` / form and retry.")
    else:
        run_params = {
            "region": region,
            "marketplace_ids": None if auto else mps,
            "sort_key": sort_key,
            "target_currency": marketplaces.REGION_CURRENCY[region],
        }
        _start_run(query.strip(), run_params)
        st.rerun()

phase = st.session_state.get("phase", "idle")

if phase in ("running", "review"):
    st_autorefresh(interval=1200, key="agent_autorefresh")

left, right = st.columns([3, 1], gap="large")

with right:
    st.markdown("#### Active agent session")
    if phase == "idle":
        st.caption("No run started yet.")
    else:
        rp = st.session_state.get("run_params", {})
        st.caption(f"Phase: **{phase}**")
        st.caption(f"Query: *{st.session_state.get('query', '')}*")
        if rp:
            st.caption(f"Region: **{marketplaces.REGION_NAMES.get(rp.get('region'), rp.get('region'))}** → target {rp.get('target_currency')}")
            st.caption(f"Ranking: *{rp.get('sort_key')}*")
            if rp.get("marketplace_ids"):
                st.caption("Sources: " + marketplaces.describe(rp.get("marketplace_ids")))
            else:
                st.caption("Sources: dynamic (auto-selected)")
        st.caption(f"Thread: `{st.session_state.get('config', {}).get('configurable', {}).get('thread_id', '—')[:8]}…`")
        selected = st.session_state.get("selected_mps") or []
        report = st.session_state.get("report")
        n_matches = len((report or {}).get("products", [])) if report else 0
        st.caption(f"Matched listings: **{n_matches}**")
        if selected:
            st.caption("Discovered: " + ", ".join(marketplaces.get(i).name for i in selected))
        if phase == "running":
            st.warning("⏳ Search & synthesis in progress…")
        st.divider()
        _render_status(st.session_state.get("nodes", []))

with left:
    if phase in ("running", "review"):
        for kind, payload in _drain(st.session_state["events"]):
            if kind == "node":
                name, update = payload
                st.session_state["nodes"].append(name)
                log.info("node update: %s %s", name, update)
                if name == "extract":
                    st.session_state["raw_results"] = update.get("raw_results", [])
                    st.session_state["raw_errors"] = update.get("raw_errors", [])
                    st.session_state["selected_mps"] = update.get("marketplace_ids", [])
                    st.session_state["report"] = None
            elif kind == "interrupt":
                st.session_state["interrupt_payload"] = payload
                st.session_state["report"] = payload.get("report", {})
                st.session_state["phase"] = "review"
                break
            elif kind == "error":
                st.session_state["run_error"] = payload
                st.session_state["phase"] = "error"
            elif kind == "done":
                st.session_state["phase"] = "done"
                st.session_state["dispatch_result"] = (payload or {}).get("dispatch_result")
                st.session_state["report"] = (payload or {}).get("report", st.session_state.get("report"))

    if st.session_state.get("run_error"):
        st.error(f"Run failed: {st.session_state['run_error']}")
        if st.button("Reset"):
            st.session_state["phase"] = "idle"
            st.rerun()

    if phase == "review" and st.session_state.get("report"):
        st.session_state.setdefault("fx_note", st.session_state["report"].get("currency_note", ""))
        _render_review(st.session_state["report"])
        with st.form("decision_form"):
            note = st.text_input("Reviewer note (optional)", key="reviewer_note")
            col_ok, col_no = st.columns(2)
            approve = col_ok.form_submit_button("✅ Approve & Dispatch", type="primary")
            reject = col_no.form_submit_button("❌ Reject")
        if approve or reject:
            decision = {"action": "approve" if approve else "reject", "note": note}
            st.session_state["gate"].submit(decision)
            st.rerun()

    elif phase == "done":
        if st.session_state.get("dispatch_result"):
            _render_dispatch(st.session_state["dispatch_result"])
        elif st.session_state.get("report"):
            st.info("❌ Rejected — the graph completed with no external action taken.")
            _render_review(st.session_state["report"])
        if st.button("New query"):
            st.session_state["phase"] = "idle"
            st.rerun()