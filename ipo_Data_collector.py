"""
IPO data collector + AI valuation enrichment.
Requires: pip install requests pandas
Set OPENAI_API_KEY in your environment before running.

Important:
- Xflot supplies IPO metadata and GMP, not complete financial valuation inputs.
- AI labels are screening labels, not investment advice.
- Only open and upcoming IPOs are sent for AI valuation, in one OpenAI API request.
- Closed/listed IPOs remain in the CSV but are marked not evaluated.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone, date

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = r"C:\Users\Kaale\Desktop\MY_Personal_project\IPO_Project\IPO_DATA\ipo_data.csv"
DASHBOARD_FILE = BASE_DIR / "ipo_dashboard_color.html"
API_URL = "https://xflot.com/api/public/market/ipos"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_LABELS = {"Undervalued", "Overvalued", "Fair"}
OPEN_STATUSES = {"open", "ongoing", "live", "currently open"}
UPCOMING_STATUSES = {"upcoming", "not yet open", "coming soon", "scheduled"}


def get_value(record, *keys):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def to_date(value):
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return value


def fetch_xflot():
    response = requests.get(
        API_URL,
        params={"limit": 200},
        headers={"User-Agent": "IPOTracker/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Xflot API returned an unsuccessful response: {payload}")
    records = payload.get("ipos", [])
    if not records:
        raise RuntimeError("Xflot returned zero IPO records.")
    return records


def normalize(records):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for item in records:
        name = get_value(item, "name", "ipoName", "companyName")
        if not name:
            continue
        rows.append({
            "ipo_name": name,
            "status": get_value(item, "status"),
            "open_date": to_date(get_value(item, "openDate", "open_date")),
            "close_date": to_date(get_value(item, "closeDate", "close_date")),
            "price_low": get_value(item, "priceMin", "price_low"),
            "price_high": get_value(item, "priceMax", "price_high"),
            "lot_size": get_value(item, "lotSize", "lot_size"),
            "issue_size_cr": get_value(item, "issueSizeCr", "issue_size_cr"),
            "gmp": get_value(item, "lastGmp", "gmp"),
            "gmp_pct": get_value(item, "lastGmpPct", "gmp_pct"),
            "subscription": get_value(item, "lastSubX", "subscription"),
            "listing_date": to_date(get_value(item, "listingDate", "listing_date")),
            "listing_price": get_value(item, "listingPrice", "listing_price"),
            "fyers_symbol": get_value(item, "symbol", "fyersSymbol"),
            "sector": get_value(item, "sector"),
            "board": get_value(item, "board"),
            "xflot_slug": get_value(item, "slug"),
            "source": "Xflot API",
            "updated_at": now,
        })
    if not rows:
        raise RuntimeError("No usable IPO records in API response.")
    df = pd.DataFrame(rows)
    # Keep distinct IPOs by API slug when available, not just company name.
    if "xflot_slug" in df:
        df["_dedupe"] = df["xflot_slug"].fillna(df["ipo_name"])
        df = df.drop_duplicates("_dedupe", keep="last").drop(columns="_dedupe")
    else:
        df = df.drop_duplicates("ipo_name", keep="last")
    return df


def is_open_or_upcoming(row, today=None):
    """Only IPOs that are open now or scheduled to open in the future qualify for AI valuation."""
    today = today or date.today()
    status = str(row.get("status") or "").strip().lower()
    open_date = pd.to_datetime(row.get("open_date"), errors="coerce")
    close_date = pd.to_datetime(row.get("close_date"), errors="coerce")
    listing_date = pd.to_datetime(row.get("listing_date"), errors="coerce")

    # Never value IPOs already listed or explicitly closed/withdrawn.
    if status in {"closed", "listed", "withdrawn", "cancelled", "canceled", "completed"}:
        return False
    if pd.notna(listing_date) and listing_date.date() <= today:
        return False
    if pd.notna(close_date) and close_date.date() < today:
        return False

    if pd.notna(open_date):
        if open_date.date() > today:
            return True
        if open_date.date() <= today and (pd.isna(close_date) or close_date.date() >= today):
            return True
    if status in OPEN_STATUSES or status in UPCOMING_STATUSES:
        return True
    # Conservative fallback for status strings such as "Open for subscription".
    if "upcoming" in status or "open" in status:
        return True
    return False


def ask_openai_once(records):
    """Send all eligible IPOs in ONE OpenAI API request."""
    if not OPENAI_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The key should be available as a system environment variable."
        )

    compact = []
    for r in records:
        compact.append({
            "ipo_key": r.get("xflot_slug") or r.get("ipo_name"),
            "ipo_name": r.get("ipo_name"),
            "status": r.get("status"),
            "open_date": r.get("open_date"),
            "close_date": r.get("close_date"),
            "sector": r.get("sector"),
            "board": r.get("board"),
            "price_low": r.get("price_low"),
            "price_high": r.get("price_high"),
            "issue_size_cr": r.get("issue_size_cr"),
            "gmp": r.get("gmp"),
            "gmp_pct": r.get("gmp_pct"),
        })

    prompt = f"""
You are a cautious Indian IPO screening assistant. Assess ONLY the supplied open or upcoming IPOs.
Return ONLY valid JSON in this exact shape:
{{"results":[{{"ipo_key":"exact input key","ipo_name":"exact input name","valuation":"Undervalued|Overvalued|Fair","reason":"short evidence-based reason, max 35 words","confidence":"Low|Medium|High"}}]}}

Rules:
- Do not use outside assumptions or invent company financials, EPS, P/E, profit, revenue, peer multiples or prospectus facts.
- GMP is unofficial short-term sentiment, NOT intrinsic valuation. Never classify as Undervalued just because GMP is positive or Overvalued just because GMP is negative.
- If no financial/peer valuation evidence is supplied, use Fair only as a neutral placeholder, confidence Low, and say clearly that financial/peer data is missing and this is not a fundamental fair-value conclusion.
- Only use Undervalued or Overvalued if the supplied facts genuinely support it. Explain the evidence and limitations briefly.
- Return exactly one result for every input record, preserve ipo_key exactly, and do not include any IPO not in the input.
Input records:
{json.dumps(compact, ensure_ascii=False)}
"""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Return valid JSON only. Never invent financial data or peer comparisons."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text[:1000]}")
    body = response.json()
    try:
        parsed = json.loads(body["choices"][0]["message"]["content"])
        return parsed.get("results", [])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse OpenAI JSON response: {body}") from exc


def enrich_open_upcoming_only(df):
    # Clear labels first so closed/listed IPOs are never shown as AI-valued from an older run.
    df["ai_valuation"] = ""
    df["ai_reason"] = "Not evaluated: IPO is not currently open or upcoming."
    df["ai_confidence"] = ""

    eligible_mask = df.apply(is_open_or_upcoming, axis=1)
    eligible = df.loc[eligible_mask].copy()
    print(f"Open/upcoming IPOs selected for AI valuation: {len(eligible)} of {len(df)}")
    if eligible.empty:
        print("No open/upcoming IPOs found; no OpenAI request was sent.")
        return df
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. No AI valuation request was sent.")

    # Exactly one request for all open/upcoming IPOs.
    print(f"Sending ONE OpenAI request for {len(eligible)} open/upcoming IPOs using {OPENAI_MODEL}...")
    results = ask_openai_once(eligible.to_dict("records"))
    by_key = {}
    for item in results:
        key = item.get("ipo_key")
        label = item.get("valuation")
        if key and label in ALLOWED_LABELS:
            by_key[str(key)] = {
                "ai_valuation": label,
                "ai_reason": str(item.get("reason", ""))[:500],
                "ai_confidence": item.get("confidence", "Low"),
            }

    missing = []
    for index, row in eligible.iterrows():
        key = str(row.get("xflot_slug") or row.get("ipo_name"))
        result = by_key.get(key)
        if result is None:
            missing.append(key)
            df.at[index, "ai_valuation"] = "Fair"
            df.at[index, "ai_reason"] = "OpenAI returned no valid result; neutral placeholder only. Verify valuation manually."
            df.at[index, "ai_confidence"] = "Low"
        else:
            for col, value in result.items():
                df.at[index, col] = value
    if missing:
        print(f"Warning: OpenAI omitted {len(missing)} result(s); these received a neutral Fair placeholder.")
    return df


def update_html_dashboard(df):
    """Embed the latest CSV records into the adjacent HTML so it opens with fresh data."""
    if not DASHBOARD_FILE.exists():
        print(f"Dashboard HTML not found beside the script: {DASHBOARD_FILE}")
        print("Download ipo_dashboard_color.html into the same folder, then run the collector again to connect them.")
        return

    html = DASHBOARD_FILE.read_text(encoding="utf-8")
    # Replace either the original empty array or data embedded by an earlier run.
    data_json = json.dumps(df.astype(object).where(pd.notna(df), None).to_dict(orient="records"), ensure_ascii=False)
    pattern = r"let data\s*=\s*(?:\[\]|JSON\.parse\('.*?'\)|\[.*?\]);const \$="
    replacement = "let data=" + data_json + ";const $="
    updated, count = re.subn(pattern, lambda _m: replacement, html, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("Could not embed IPO records: dashboard data declaration was not found.")
    DASHBOARD_FILE.write_text(updated, encoding="utf-8")
    print(f"Updated connected HTML dashboard: {DASHBOARD_FILE}")


def main():
    fresh = normalize(fetch_xflot())
    print(f"Xflot records after API-slug deduplication: {len(fresh)}")
    fresh = enrich_open_upcoming_only(fresh)
    fresh.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(fresh)} IPO records to: {CSV_FILE}")
    update_html_dashboard(fresh)
    eligible = fresh.apply(is_open_or_upcoming, axis=1)
    print(f"Open/upcoming IPOs evaluated: {int(eligible.sum())}")
    print("\nAI valuation counts for open/upcoming IPOs:")
    if eligible.any():
        print(fresh.loc[eligible, "ai_valuation"].value_counts(dropna=False).to_string())
    else:
        print("No eligible IPOs today.")
    print("\nReminder: AI labels are preliminary screening only, not investment advice.")


if __name__ == "__main__":
    main()
