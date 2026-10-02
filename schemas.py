from __future__ import annotations

import os
import urllib.request
from typing import Callable

from pydantic import BaseModel, Field, field_validator, model_validator

import marketplaces

KNOWN_CURRENCIES = ("PKR", "USD", "EUR", "GBP", "CNY", "AED", "SAR", "INR", "NGN", "KES")

# Approximate units per 1 USD; used as static fallback when live FX is off/unreachable.
FIXED_RATES_PER_USD = {
    "USD": 1.0,
    "PKR": 278.5,
    "EUR": 0.92,
    "GBP": 0.79,
    "CNY": 7.2,
    "AED": 3.67,
    "SAR": 3.75,
    "INR": 83.5,
    "NGN": 1600.0,
    "KES": 128.0,
}

_LIVE_FX_URL = "https://open.er-api.com/v6/latest/USD"

_CURRENCY_ALIASES = {
    "RS": "PKR", "RS.": "PKR", "PKR": "PKR", "RUPEE": "PKR", "RUPEE.": "PKR", "RPS": "PKR",
    "US$": "USD", "USD": "USD", "$": "USD", "DOLLAR": "USD", "US DOLLAR": "USD",
    "EUR": "EUR", "EURO": "EUR", "€": "EUR",
    "GBP": "GBP", "£": "GBP", "POUND": "GBP",
    "CNY": "CNY", "RMB": "CNY", "YUAN": "CNY", "¥": "CNY",
    "AED": "AED", "DIRHAM": "AED",
    "SAR": "SAR", "RIYAL": "SAR",
    "INR": "INR", "₹": "INR", "RUPEE.": "INR",
    "NGN": "NGN", "NAIRA": "NGN", "₦": "NGN",
    "KES": "KES", "SHILLING": "KES", "KSH": "KES",
}

SORT_KEYS = ("price", "sales", "stars", "reviews")


def _sort_desc(key: str) -> bool:
    return key in ("sales", "stars", "reviews")


SORTERS: dict[str, Callable[[ProductListing], float]] = {}  # filled after ProductListing


class CurrencyStandardizer:
    """Cross-currency normalization to a target ('local') currency.

    Uses live rates (open.er-api.com) when LIVE_FX=1 (default), falling back to
    FIXED_RATES_PER_USD on any failure. Convert every listing into the user's
    home currency so distant markets are directly comparable.
    """

    _rates_per_usd: dict[str, float] = dict(FIXED_RATES_PER_USD)
    _target: str = os.getenv("TARGET_CURRENCY", "PKR").upper()
    _live: bool = os.getenv("LIVE_FX", "1").strip() not in ("0", "false", "")
    _refreshed: bool = False
    _note: str = "fixed rates"

    # --- lifecycle ------------------------------------------------------ #
    @classmethod
    def reload(cls) -> None:
        cls._target = os.getenv("TARGET_CURRENCY", "PKR").upper()
        if cls._target not in KNOWN_CURRENCIES:
            raise ValueError(f"Unsupported TARGET_CURRENCY={cls._target}")
        cls._live = os.getenv("LIVE_FX", "1").strip() not in ("0", "false", "")
        cls._rates_per_usd = dict(FIXED_RATES_PER_USD)
        cls._note = "fixed rates"
        cls._refreshed = False

    @classmethod
    def set_target(cls, currency: str) -> None:
        currency = currency.strip().upper()
        if currency not in KNOWN_CURRENCIES:
            raise ValueError(f"Unsupported target currency: {currency}")
        cls._target = currency

    @classmethod
    def _ensure_rates(cls) -> None:
        if cls._refreshed or not cls._live:
            return
        cls._refreshed = True
        try:
            with urllib.request.urlopen(_LIVE_FX_URL, timeout=8) as resp:
                data = resp.read()
            import json

            rates = json.loads(data).get("rates", {})
            known = {c: rates[c] for c in KNOWN_CURRENCIES if rates.get(c)}
            if known and len(known) > 1:
                cls._rates_per_usd = known
                cls._note = "live (open.er-api.com)"
        except Exception:  # noqa: BLE001 - degrade to fixed, never block a run
            cls._note = "fixed rates (live FX failed)"

    # --- queries -------------------------------------------------------- #
    @classmethod
    def target(cls) -> str:
        return cls._target

    @classmethod
    def note(cls) -> str:
        cls._ensure_rates()
        return cls._note

    @classmethod
    def per_usd(cls, currency: str) -> float:
        cls._ensure_rates()
        if currency not in cls._rates_per_usd:
            raise ValueError(f"No rate known for currency {currency!r}")
        return cls._rates_per_usd[currency]

    @classmethod
    def convert(cls, amount: float, from_currency: str) -> float:
        cls._ensure_rates()
        if from_currency == cls._target:
            return round(amount, 2)
        return round(amount * cls.per_usd(cls._target) / cls.per_usd(from_currency), 2)

    @classmethod
    def display_rate(cls, from_currency: str = "USD") -> str:
        cls._ensure_rates()
        if from_currency == cls._target:
            return f"Prices already in {cls._target}. Rate source: {cls._note}."
        try:
            one = cls.convert(1.0, from_currency)
        except ValueError:
            one = None
        base = f"1 {from_currency} = {one:,.2f} {cls._target}" if one is not None else f"No {from_currency} rate"
        return f"{base} · source: {cls._note}"


class ProductListing(BaseModel):
    model: str = Field(description="Normalized product model / SKU identifier")
    title: str = Field(description="Listing title as shown by the seller")
    source: str = Field(description="Marketplace id, e.g. daraz_pk / amazon_de / aliexpress")
    store: str = Field(default="", description="Seller / store name if detectable")
    currency: str = Field(description="Original listing currency")
    raw_price: float = Field(ge=0, description="Item price in the original currency")
    shipping: float = Field(default=0.0, ge=0, description="Shipping cost in the original currency")
    sales: int = Field(default=0, ge=0, description="Units sold (from listing, if visible)")
    rating: float = Field(default=0.0, ge=0, le=5, description="Seller/listing stars 0-5")
    reviews: int = Field(default=0, ge=0, description="Review count, if visible")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Extraction confidence 0-1")
    url: str = Field(description="Direct product URL")
    price_local: float = Field(default=0.0, description="Price converted to the local currency (auto)")
    landed_local: float = Field(default=0.0, description="Price + shipping, converted to local (auto)")

    @field_validator("source")
    @classmethod
    def _valid_source(cls, v: str) -> str:
        v = v.strip().lower().replace(".", "_")
        aliases = {"daraz": "daraz_pk", "daraz.pk": "daraz_pk", "aliexpress.com": "aliexpress"}
        v = aliases.get(v, v)
        if v not in marketplaces.MARKETPLACES:
            raise ValueError(f"unknown source/marketplace {v!r}")
        return v

    @field_validator("currency")
    @classmethod
    def _valid_currency(cls, v: str) -> str:
        norm = str(v).strip()
        upper = norm.upper()
        alias = _CURRENCY_ALIASES.get(upper, _CURRENCY_ALIASES.get(norm.strip().rstrip("."), _CURRENCY_ALIASES.get(norm)))
        if alias:
            return alias
        if upper in KNOWN_CURRENCIES:
            return upper
        raise ValueError(f"unsupported currency {v!r}")

    @model_validator(mode="before")
    @classmethod
    def _normalize_in(cls, data):
        if isinstance(data, dict):
            data["currency"] = data.get("currency", "")
            data["source"] = data.get("source", "")
        return data

    @model_validator(mode="after")
    def _enforce_local(self) -> "ProductListing":
        self.price_local = CurrencyStandardizer.convert(self.raw_price, self.currency)
        self.landed_local = self.price_local + CurrencyStandardizer.convert(self.shipping, self.currency)
        self.landed_local = round(self.landed_local, 2)
        return self

    @property
    def marketplace_name(self) -> str:
        try:
            return marketplaces.get(self.source).name
        except ValueError:
            return self.source

    @property
    def country(self) -> str:
        try:
            return marketplaces.get(self.source).country
        except ValueError:
            return ""

    def as_payload(self) -> dict:
        return {
            "model": self.model,
            "title": self.title,
            "source": self.source,
            "marketplace": self.marketplace_name,
            "country": self.country,
            "store": self.store,
            "currency": self.currency,
            "raw_price": self.raw_price,
            "shipping": self.shipping,
            "sales": self.sales,
            "rating": self.rating,
            "reviews": self.reviews,
            "confidence": self.confidence,
            "url": self.url,
            "price_local": self.price_local,
            "landed_local": self.landed_local,
        }


SORTERS.update(
    {
        "price": lambda p: p.price_local,
        "landed": lambda p: p.landed_local,
        "sales": lambda p: p.sales,
        "stars": lambda p: p.rating,
        "reviews": lambda p: p.reviews,
    }
)


def _sort_key(key: str) -> str:
    key = key.strip().lower()
    return key if key in SORTERS else "price"


class ComparisonReport(BaseModel):
    query: str = Field(description="Original user product query")
    as_of: str = Field(description="ISO timestamp of report synthesis")
    region: str = Field(default="PK", description="User region (ISO2)")
    target_currency: str = Field(default="PKR", description="Comparison currency")
    sort_key: str = Field(default="price", description="How products are ranked for display")
    currency_note: str = Field(
        default_factory=lambda: CurrencyStandardizer.display_rate(), description="FX explanation"
    )
    products: list[ProductListing] = Field(default_factory=list)
    best_deal: ProductListing | None = Field(default=None, description="Cheapest verified match in local currency")
    no_match: bool = Field(default=False, description="True when nothing usable was extracted")
    summary: str = Field(default="", description="Human-readable one-liner for the review card")

    @model_validator(mode="after")
    def _finalize(self) -> "ComparisonReport":
        # Drop listings where no price could be extracted — they're useless for comparison.
        priced = [p for p in self.products if p and p.raw_price and p.raw_price > 0]
        if priced and len(priced) < len(self.products):
            self.products = priced
        if self.products:
            self.products = sorted(self.products, key=SORTERS[_sort_key(self.sort_key)], reverse=_sort_desc(self.sort_key))
        if not self.products:
            self.no_match = True
            self.summary = "No usable priced listings were extracted from the sources."
            self.best_deal = None
        else:
            self.best_deal = min((p for p in self.products), key=lambda p: p.price_local)
            best = self.best_deal
            self.no_match = False
            self.summary = (
                f"[{self.sort_key}] {self.products[0].title} on {self.products[0].marketplace_name} leads. "
                f"Cheapest: {best.title} on {best.marketplace_name} at {self.target_currency} "
                f"{best.price_local:,.2f} ({len(self.products)} priced listings compared)."
            )
        return self


class Decision(BaseModel):
    action: str = Field(description="Human decision: 'approve' or 'reject'")
    note: str = Field(default="", description="Optional reviewer note")

    @field_validator("action")
    @classmethod
    def _valid_action(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("approve", "reject"):
            raise ValueError(f"action must be 'approve' or 'reject', got {v!r}")
        return v


class DispatchResult(BaseModel):
    action: str = Field(description="Action taken")
    status: str = Field(description="'dispatched', 'dry_run' or 'skipped'")
    target: str = Field(default="", description="Where the action went (webhook URL or log file)")
    detail: str = Field(default="", description="Payload or error message")