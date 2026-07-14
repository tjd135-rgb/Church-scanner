"""Verify the pipeline caught the four known-good parcels.

Match strategy:
- If parcel_id is given in the config, try that first (digits-only compare
  because some sources dash-format the folio, others don't).
- Otherwise, fall back to fuzzy address match: substring of the config's
  `address_contains` in the parcel's canonical situs address, plus city.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ValidationHit:
    name: str
    matched: bool
    matched_parcel_id: str | None
    matched_address: str | None
    reason: str


def _digits(s: object) -> str:
    return re.sub(r"\D", "", str(s or ""))


def validate(df: pd.DataFrame, expected: list[dict]) -> list[ValidationHit]:
    hits: list[ValidationHit] = []
    if df.empty:
        return [
            ValidationHit(e["name"], False, None, None,
                          "no parcels returned at all")
            for e in expected
        ]

    df_pid = df["parcel_id"].map(_digits) if "parcel_id" in df.columns else pd.Series([], dtype=str)
    for e in expected:
        want_pid = _digits(e.get("parcel_id")) if e.get("parcel_id") else ""
        want_addr = str(e.get("address_contains", "")).upper()
        want_city = str(e.get("city", "")).upper()
        hit: pd.DataFrame | None = None
        reason = ""
        if want_pid:
            m = df[df_pid == want_pid]
            if not m.empty:
                hit = m
                reason = f"matched by parcel_id={want_pid}"
        if hit is None and want_addr:
            if "situs_addr" not in df.columns:
                continue
            addrs = df["situs_addr"].astype("string").str.upper().fillna("")
            addr_mask = addrs.str.contains(re.escape(want_addr), regex=True)
            if want_city and "situs_city" in df.columns:
                cities = df["situs_city"].astype("string").str.upper().fillna("")
                city_mask = cities.str.contains(want_city)
                full_mask = addr_mask & city_mask
            else:
                full_mask = addr_mask
            m = df[full_mask]
            if not m.empty:
                hit = m
                reason = f"matched by address substring '{want_addr}'"
        if hit is not None:
            row = hit.iloc[0]
            hits.append(ValidationHit(
                name=e["name"],
                matched=True,
                matched_parcel_id=str(row.get("parcel_id") or ""),
                matched_address=str(row.get("situs_addr") or ""),
                reason=reason,
            ))
        else:
            hits.append(ValidationHit(
                name=e["name"],
                matched=False,
                matched_parcel_id=None,
                matched_address=None,
                reason=(
                    f"NOT FOUND. tried parcel_id={want_pid or 'n/a'}, "
                    f"address substring '{want_addr}' in {want_city or 'any'}"
                ),
            ))
    return hits


def report(hits: list[ValidationHit]) -> str:
    lines = ["=== VALIDATION AGAINST KNOWN PARCELS ==="]
    ok = sum(1 for h in hits if h.matched)
    for h in hits:
        marker = "[OK] " if h.matched else "[MISS]"
        if h.matched:
            lines.append(
                f"{marker} {h.name}: {h.matched_address} "
                f"(parcel {h.matched_parcel_id}) — {h.reason}"
            )
        else:
            lines.append(f"{marker} {h.name}: {h.reason}")
    lines.append(f"\n{ok}/{len(hits)} known parcels found.")
    return "\n".join(lines)
