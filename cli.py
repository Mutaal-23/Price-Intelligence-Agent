#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import agent_graph
import marketplaces
from langgraph.checkpoint.memory import MemorySaver
from schemas import ComparisonReport, ProductListing


def _fixture_graph() -> None:
    from datetime import datetime, timezone

    def fixture_extract(state):
        rows = [
            {
                "source": "daraz_pk", "marketplace": "Daraz", "country": "Pakistan", "currency": "PKR",
                "title": "Original Samsung EB-BG965ABE Battery for Galaxy S9 Plus 3500mAh",
                "url": "https://www.daraz.pk/products/samsung-eb-bg965abe-12345",
                "snippet": "Original Samsung EB-BG965ABE battery for Galaxy S9+ 3500mAh. Price Rs. 4,499. "
                           "180 sold. Free delivery.",
            },
            {
                "source": "aliexpress", "marketplace": "AliExpress", "country": "Global", "currency": "USD",
                "title": "EB-BG965ABE Battery Replacement for Samsung S9 Plus",
                "url": "https://www.aliexpress.com/item/3256800000000.html",
                "snippet": "EB-BG965ABE battery 3500mAh for Samsung Galaxy S9 Plus, US $ 15.99, 4.7, "
                           "330 sold, free returns.",
            },
        ]
        return {"raw_results": rows, "marketplace_ids": ["daraz_pk", "aliexpress"]}

    def fixture_synthesize(state):
        products = [
            ProductListing(
                model="EB-BG965ABE", title="Original Samsung EB-BG965ABE Battery for Galaxy S9 Plus 3500mAh",
                source="daraz_pk", store="Official Samsung Store", currency="PKR", raw_price=4499.0,
                sales=180, url="https://www.daraz.pk/products/samsung-eb-bg965abe-12345", confidence=0.98,
            ),
            ProductListing(
                model="EB-BG965ABE", title="EB-BG965ABE Battery Replacement for Samsung S9 Plus",
                source="aliexpress", store="Kingstore", currency="USD", raw_price=15.99,
                sales=330, rating=4.7, url="https://www.aliexpress.com/item/3256800000000.html", confidence=0.95,
            ),
        ]
        report = ComparisonReport(
            query=state["query"],
            as_of=datetime.now(timezone.utc).isoformat(),
            region=state.get("region", "PK"),
            target_currency=state.get("target_currency", "PKR"),
            sort_key=state.get("sort_key", "price"),
            products=products,
        )
        return {"report": report.model_dump(mode="json")}

    agent_graph.GRAPH = agent_graph.build_graph(
        extract_fn=fixture_extract, synthesize_fn=fixture_synthesize, checkpointer=MemorySaver()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless runner for the price intelligence graph.")
    parser.add_argument("query", help="Product name / SKU to analyze")
    parser.add_argument("--region", default="PK", choices=sorted(marketplaces.REGION_DEFAULTS),
                        help="User region — selects the default marketplace set + local currency")
    parser.add_argument("--marketplace", dest="mp", action="append", metavar="ID",
                        help="Extra/override marketplace id (repeatable; merged with region defaults)")
    parser.add_argument("--sort", default="price", choices=list(__import__("schemas").SORTERS),
                        help="Ranking for the report")
    parser.add_argument("--target", default=None, metavar="CUR", help="Comparison currency (default = region currency)")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-resume the interrupt with approve")
    parser.add_argument("--auto-reject", action="store_true", help="Auto-resume the interrupt with reject")
    parser.add_argument("--fixture", action="store_true", help="Use offline sample data (no Tavily/LLM keys)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print final state as JSON")
    args = parser.parse_args()

    auto_resume = None
    if args.auto_approve:
        auto_resume = {"action": "approve", "note": "auto-approved via CLI"}
    if args.auto_reject:
        auto_resume = {"action": "reject", "note": "auto-rejected via CLI"}

    if args.fixture:
        _fixture_graph()

    mps = None
    if args.mp:
        mps = list(marketplaces.default_for(args.region))
        for i in args.mp:
            marketplaces.get(i)
            if i not in mps:
                mps.append(i)

    final = None
    for kind, payload in agent_graph.run_stream(
        args.query, auto_resume=auto_resume, region=args.region,
        marketplace_ids=mps, sort_key=args.sort, target_currency=args.target,
    ):
        if kind == "node":
            print(f"\n[node:{payload[0]}]")
            print(json.dumps(payload[1], indent=2, default=str))
        elif kind == "interrupt":
            print("\n[interrupt] graph PAUSED for human review")
            src_label = marketplaces.describe(mps) if mps else "dynamic (auto-selected)"
            print(f"sources: {src_label}")
            report = payload.get("report", {})
            for p in report.get("products", []):
                print(json.dumps(p, indent=2, default=str))
            if auto_resume is None:
                print("\n  -> no --auto-* flag given; thread left paused on MemorySaver")
                return 0
            print(f"  -> auto-resuming with {json.dumps(auto_resume)}")
        elif kind == "error":
            print(f"\n[error] {payload}", file=sys.stderr)
            return 1
        elif kind == "done":
            final = payload

    assert final is not None, "graph did not reach a terminal state"
    print("\n=== FINAL STATE ===")
    if args.as_json:
        print(json.dumps(final, indent=2, default=str))
    else:
        report = final.get("report") or {}
        print("query:          ", final.get("query"))
        print("region:         ", final.get("region"), "| target:", report.get("target_currency"), "| sort:", report.get("sort_key"))
        print("sources:        ", marketplaces.describe(final.get("marketplace_ids", [])))
        print("currency_note:  ", report.get("currency_note"))
        print("summary:        ", report.get("summary"))
        for p in report.get("products", []):
            mp = p.get("marketplace", p.get("source"))
            print(
                f" - [{mp:>12}] {p.get('model'):<20} {report.get('target_currency')} {p.get('price_local'):>10,.2f} "
                f"(landed {p.get('landed_local'):,.2f}) sales={p.get('sales')} stars={p.get('rating')}  {p.get('title')[:52]}"
            )
        print("dispatch_result:", json.dumps(final.get("dispatch_result"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())