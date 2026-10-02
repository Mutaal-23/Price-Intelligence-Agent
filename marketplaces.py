from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Marketplace:
    id: str
    name: str
    domain: str
    region: str
    country: str
    currency: str
    url_filters: tuple[str, ...] = ()
    notes: str = ""


MARKETPLACES: dict[str, Marketplace] = {
    m.id: m
    for m in [
        Marketplace("daraz_pk", "Daraz", "daraz.pk", "PK", "Pakistan", "PKR", ("/products/",)),
        Marketplace("aliexpress", "AliExpress", "aliexpress.com", "CN", "Global", "USD",
                    ("/item/", "/i/"), notes="Cross-border import"),
        Marketplace("temu", "Temu", "temu.com", "CN", "China", "USD", ("/goods/",)),
        Marketplace("amazon_de", "Amazon DE", "amazon.de", "DE", "Germany", "EUR", ("/dp/", "/gp/")),
        Marketplace("amazon_com", "Amazon US", "amazon.com", "US", "United States", "USD", ("/dp/", "/gp/")),
        Marketplace("amazon_uk", "Amazon UK", "amazon.co.uk", "UK", "United Kingdom", "GBP", ("/dp/", "/gp/")),
        Marketplace("amazon_ae", "Amazon AE", "amazon.ae", "AE", "UAE", "AED", ("/dp/", "/gp/")),
        Marketplace("amazon_sa", "Amazon SA", "amazon.sa", "SA", "Saudi Arabia", "SAR", ("/dp/", "/gp/")),
        Marketplace("amazon_in", "Amazon IN", "amazon.in", "IN", "India", "INR", ("/dp/", "/gp/")),
        Marketplace("flipkart", "Flipkart", "flipkart.com", "IN", "India", "INR", ("/p/",)),
        Marketplace("ebay", "eBay", "ebay.com", "US", "Global", "USD", ("/itm/",), notes="Auction/marketplace"),
        Marketplace("noon", "Noon", "noon.com", "AE", "UAE/KSA", "AED", ("/noon/", "/product")),
        Marketplace("jumia_ng", "Jumia NG", "jumia.com.ng", "NG", "Nigeria", "NGN", ("/product/",)),
        Marketplace("jumia_ke", "Jumia KE", "jumia.co.ke", "KE", "Kenya", "KES", ("/product/",)),
    ]
}

# Default marketplace mix per user region. First entries = local, then cross-border.
REGION_DEFAULTS: dict[str, list[str]] = {
    "PK": ["daraz_pk", "aliexpress"],
    "DE": ["amazon_de", "ebay", "aliexpress"],
    "US": ["amazon_com", "ebay"],
    "UK": ["amazon_uk", "ebay"],
    "CN": ["temu", "aliexpress"],
    "NG": ["jumia_ng", "aliexpress"],
    "KE": ["jumia_ke", "aliexpress"],
    "AE": ["amazon_ae", "noon", "aliexpress"],
    "SA": ["amazon_sa", "noon", "aliexpress"],
    "IN": ["flipkart", "amazon_in"],
}

# Home currency per region (the comparison "local" target).
REGION_CURRENCY: dict[str, str] = {
    "PK": "PKR", "DE": "EUR", "US": "USD", "UK": "GBP", "CN": "CNY",
    "NG": "NGN", "KE": "KES", "AE": "AED", "SA": "SAR", "IN": "INR",
}

REGION_NAMES: dict[str, str] = {
    "PK": "🇵🇰 Pakistan", "DE": "🇩🇪 Germany", "US": "🇺🇸 USA", "UK": "🇬🇧 United Kingdom",
    "CN": "🇨🇳 China", "NG": "🇳🇬 Nigeria", "KE": "🇰🇪 Kenya", "AE": "🇦🇪 UAE",
    "SA": "🇸🇦 Saudi Arabia", "IN": "🇮🇳 India",
}


def get(mp_id: str) -> Marketplace:
    try:
        return MARKETPLACES[mp_id]
    except KeyError:
        raise ValueError(f"Unknown marketplace: {mp_id!r}. Known: {sorted(MARKETPLACES)}") from None


def default_for(region: str) -> list[str]:
    if region not in REGION_DEFAULTS:
        raise ValueError(f"Unknown region {region!r}; choose from {sorted(REGION_DEFAULTS)}")
    return REGION_DEFAULTS[region]


def ordered_for(region: str) -> list[str]:
    """All marketplace ids, ordered by relevance for a region:
    local/default shops first, then cross-border + the rest.
    Used for dynamic auto-selection — we scan widely and let the data decide."""
    if region not in REGION_DEFAULTS:
        raise ValueError(f"Unknown region {region!r}; choose from {sorted(REGION_DEFAULTS)}")
    base = list(REGION_DEFAULTS[region])
    return base + [i for i in MARKETPLACES if i not in base]


def all_ids() -> list[str]:
    return sorted(MARKETPLACES)


def is_product_page(mp: Marketplace, url: str) -> bool:
    if not mp.url_filters:
        return True
    return any(frag in url for frag in mp.url_filters)


def describe(mp_ids: list[str]) -> str:
    return ", ".join(
        f"{get(i).name} ({get(i).country}, {get(i).currency})" for i in mp_ids
    )