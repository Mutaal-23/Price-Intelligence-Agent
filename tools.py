from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from tavily import TavilyClient  # noqa: E402

import marketplaces  # noqa: E402

TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "8"))
TAVILY_DEPTH = os.getenv("TAVILY_DEPTH", "advanced").strip().lower()
TAVILY_MAX_MARKETPLACES = int(os.getenv("TAVILY_MAX_MARKETPLACES", "6"))
TAVILY_SCAN_MARKETPLACES = int(os.getenv("TAVILY_SCAN_MARKETPLACES", "12"))

_PLACEHOLDER_MARKERS = ("your-", "your_task", "changeme")


def resolve_tavily_key() -> str | None:
    """Return a usable Tavily key, honoring TAVILY_API or TAVILY_API_KEY."""
    for name in ("TAVILY_API_KEY", "TAVILY_API"):
        value = (os.getenv(name) or "").strip()
        if value and not any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
            return value
    return None


def _client() -> TavilyClient:
    key = resolve_tavily_key()
    if not key:
        raise RuntimeError(
            "No valid Tavily key found. Set TAVILY_API_KEY in .env (or the legacy TAVILY_API "
            "var) and restart. Current TAVILY_API_KEY is a placeholder."
        )
    return TavilyClient(api_key=key)


def _search_one(mp: marketplaces.Marketplace, query: str, max_results: int) -> list[dict]:
    augmented = f'{query.strip()} site:{mp.domain} buy price'
    try:
        resp = _client().search(
            query=augmented,
            search_depth=TAVILY_DEPTH,
            max_results=max_results,
            include_raw_content=False,
            include_answer=False,
        )
    except Exception as exc:  # keep one failing source from killing the whole run
        return [{"error": str(exc), "source": mp.id, "query": query}]

    rows: list[dict] = []
    for item in resp.get("results", []):
        url = item.get("url", "")
        if not marketplaces.is_product_page(mp, url):
            continue  # drop category/tag/landing pages — not a real listing
        rows.append(
            {
                "source": mp.id,
                "marketplace": mp.name,
                "country": mp.country,
                "currency": mp.currency,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("content", ""),
                "scroll_id": item.get("id"),
            }
        )
    return rows


def search_marketplace(mp_id: str, query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    return _search_one(marketplaces.get(mp_id), query, max_results)


@dataclass
class RawExtraction:
    query: str
    marketplace_ids: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def hits_for(self, mp_id: str) -> list[dict]:
        return [r for r in self.results if r.get("source") == mp_id]


def extract_sources(query: str, mp_ids: list[str], max_results: int = TAVILY_MAX_RESULTS) -> RawExtraction:
    """Fetch the given marketplaces in parallel; only product pages are kept."""
    with ThreadPoolExecutor(max_workers=max(2, len(mp_ids))) as pool:
        futures = {pool.submit(_search_one, marketplaces.get(i), query, max_results): i for i in mp_ids}
        payload: list[list[dict]] = [f.result() for f in futures]

    errors = [r for batch in payload for r in batch if "error" in r]
    results = [r for batch in payload for r in batch if "error" not in r]
    return RawExtraction(query=query, marketplace_ids=mp_ids, results=results, errors=errors)


def select_best_marketplaces(region: str, results: list[dict], limit: int | None = None) -> list[str]:
    """Dynamic marketplace selection — keep the sites that actually yielded product
    pages, ordered by region preference (local first) then hit count. Ties go local.

    Returns the empty-friendly fallback of the region defaults when nothing matched.
    """
    if limit is None:
        limit = TAVILY_MAX_MARKETPLACES
    order = {mid: pos for pos, mid in enumerate(marketplaces.ordered_for(region))}
    counts = Counter(r.get("source") for r in results if not r.get("error") and r.get("source"))
    if not counts:
        return marketplaces.default_for(region)
    ranked = sorted(counts, key=lambda i: (order.get(i, 99), -counts[i]))
    return ranked[:limit]