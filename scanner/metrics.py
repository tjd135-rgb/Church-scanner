"""Redevelopment metrics and composite score.

All four components are normalized to a 0..100 sub-score:

  acreage:            0 at 0 acres, 100 at score_caps.acreage_ac_max.
  land_utilization:   scored on 1 - LUR clamped to lur_zero_target.
                      LUR = building SF / lot SF.
                      LUR <= lur_zero_target -> 100 (very under-built).
                      LUR >= 1.0             -> 0.
  land_value_share:   scored on LVS / land_value_share_max, clamped to 100.
                      LVS = land_value / total_value.
  hold_period:        scored on years_since_last_sale / hold_period_years_max.
                      Parcels with no sale on record are treated as
                      "very long hold" and given the full 100.

Composite = weighted sum, always in [0, 100].

The output DataFrame keeps every sub-score alongside the composite so a
reader can trace exactly why any given row scores where it does.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_metrics(
    df: pd.DataFrame,
    weights: dict[str, float],
    caps: dict[str, float],
    today: datetime | None = None,
) -> pd.DataFrame:
    today = today or datetime.utcnow()

    lot_sf = pd.to_numeric(df.get("lot_sf"), errors="coerce")
    lot_ac = pd.to_numeric(df.get("lot_ac"), errors="coerce")
    bldg_sf = pd.to_numeric(df.get("bldg_sf"), errors="coerce")
    land_val = pd.to_numeric(df.get("land_val"), errors="coerce")
    total_val = pd.to_numeric(df.get("total_val"), errors="coerce")
    sale_date = pd.to_datetime(df.get("sale_date"), errors="coerce")

    # --- Component: LUR ------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        lur = np.where(
            (lot_sf > 0) & bldg_sf.notna(),
            bldg_sf / lot_sf,
            np.nan,
        )
    df["land_utilization_ratio"] = lur

    # --- Component: LVS ------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        lvs = np.where(
            (total_val > 0) & land_val.notna(),
            land_val / total_val,
            np.nan,
        )
    df["land_value_share"] = lvs

    # --- Component: hold period ---------------------------------------
    hold_years = (today - sale_date).dt.days / 365.25
    df["hold_period_years"] = hold_years

    # --- Sub-scores 0..100 --------------------------------------------
    acreage_cap = caps.get("acreage_ac_max", 10.0)
    lur_target = caps.get("lur_zero_target", 0.05)
    lvs_cap = caps.get("land_value_share_max", 0.9)
    hold_cap = caps.get("hold_period_years_max", 40)

    df["score_acreage"] = _clip01(lot_ac.fillna(0) / acreage_cap) * 100

    lur_sub = np.where(
        np.isnan(lur),
        50.0,  # missing -> neutral so we don't distort the composite
        np.clip(
            (1.0 - (lur - lur_target) / (1.0 - lur_target)) * 100,
            0.0,
            100.0,
        ),
    )
    df["score_lur_inverse"] = lur_sub

    lvs_sub = np.where(
        np.isnan(lvs),
        50.0,
        np.clip((lvs / lvs_cap) * 100, 0.0, 100.0),
    )
    df["score_land_value_share"] = lvs_sub

    hold_sub = np.where(
        hold_years.isna(),
        100.0,  # no sale on record treated as long hold
        np.clip((hold_years / hold_cap) * 100, 0.0, 100.0),
    )
    df["score_hold_period"] = hold_sub

    composite = (
        weights["acreage"] * df["score_acreage"]
        + weights["lur_inverse"] * df["score_lur_inverse"]
        + weights["land_value_share"] * df["score_land_value_share"]
        + weights["hold_period"] * df["score_hold_period"]
    )
    df["redevelopment_score"] = composite.round(1)
    return df


def weight_documentation(weights: dict[str, float],
                        caps: dict[str, float]) -> list[str]:
    """Human-readable lines describing the scoring model for the summary tab."""
    return [
        "Redevelopment score = weighted sum of four 0..100 sub-scores.",
        f"  acreage weight            = {weights['acreage']:.2f}   "
        f"(saturates at {caps.get('acreage_ac_max')} acres = 100)",
        f"  land_utilization weight   = {weights['lur_inverse']:.2f}   "
        f"(LUR = bldg SF / lot SF; <= {caps.get('lur_zero_target')} -> 100)",
        f"  land_value_share weight   = {weights['land_value_share']:.2f}   "
        f"(land_val / total_val; >= {caps.get('land_value_share_max')} -> 100)",
        f"  hold_period weight        = {weights['hold_period']:.2f}   "
        f"(years since last sale; >= {caps.get('hold_period_years_max')} -> 100; "
        f"no sale on record = 100)",
        "Missing LUR or LVS values contribute a neutral 50 sub-score so the "
        "composite doesn't unfairly bury or reward data gaps.",
    ]
