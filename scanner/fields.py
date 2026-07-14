"""Field-alias mapping.

Different Florida parcel services expose the same tax-roll data under
different column names. This module keeps a single canonical schema and a
generous list of aliases that gets resolved at runtime by inspecting each
layer's `?f=json` metadata. The rest of the pipeline only sees the
canonical names.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical field names used throughout the pipeline.
CANONICAL = [
    "parcel_id",
    "county",
    "owner_name",
    "owner_addr1",
    "owner_addr2",
    "owner_city",
    "owner_state",
    "owner_zip",
    "situs_addr",
    "situs_city",
    "situs_zip",
    "dor_uc",
    "zoning",
    "land_val",
    "bldg_val",
    "total_val",
    "lot_sf",
    "lot_ac",
    "bldg_sf",
    "year_built",
    "sale_date",
    "sale_price",
]

# Alias table. Keys are canonical; values are lists of candidate source
# field names in priority order (case-insensitive match). The list mixes
# the FGDL statewide schema, Broward BCPA, and Palm Beach PAPA
# conventions — the resolver picks whichever the target layer actually
# exposes.
ALIASES: dict[str, list[str]] = {
    "parcel_id": [
        "PARCELNO", "PARCEL_ID", "PARCELID", "FOLIO", "FOLIONUM",
        "PARCEL_NUM", "PCN", "PARCEL", "PIN",
    ],
    "county": ["CO_NO", "COUNTY", "COUNTY_NAME", "CNTY", "CNTY_NAME"],
    "owner_name": [
        "OWN_NAME", "OWNER_NAME", "OWNER", "OWN_NAME1", "OWNERNAME",
        "OWNER1", "OWNER_1", "OWNER_NAM",
    ],
    "owner_addr1": [
        "OWN_ADDR1", "OWN_ADDR_1", "OWNER_ADDR", "OWNER_ADDR1",
        "MAIL_ADDR1", "MAILING_ADDRESS", "OWN_ADDR",
    ],
    "owner_addr2": ["OWN_ADDR2", "OWN_ADDR_2", "MAIL_ADDR2"],
    "owner_city": ["OWN_CITY", "OWNER_CITY", "MAIL_CITY"],
    "owner_state": ["OWN_STATE", "OWNER_STATE", "MAIL_STATE"],
    "owner_zip": ["OWN_ZIPCD", "OWN_ZIP", "OWNER_ZIP", "MAIL_ZIP"],
    "situs_addr": [
        "PHY_ADDR1", "SITE_ADDR", "SITUS_ADDRESS", "SITUS_ADDR",
        "SITE_ADDRESS", "PROPERTY_ADDRESS", "ADDRESS", "STREET_ADDR",
        "SITUSADDR",
    ],
    "situs_city": [
        "PHY_CITY", "SITE_CITY", "SITUS_CITY", "PROPERTY_CITY", "CITY",
    ],
    "situs_zip": ["PHY_ZIPCD", "SITE_ZIP", "SITUS_ZIP", "ZIP"],
    "dor_uc": [
        # FGIO / FGDL statewide layer uses the "01" suffix (up to 6 codes per parcel).
        # DOR_UC (no suffix) is the legacy FGDL name; keep it as a fallback.
        "DOR_UC01", "DORUC01", "LNDUSE_01", "LNDUSE01", "LNDUSE_1",
        "DOR_UC", "DORUC", "USE_CODE", "USECODE", "LAND_USE",
        "LANDUSE", "LAND_USE_CODE", "PROP_USE",
    ],
    "zoning": ["ZONING", "ZONE", "ZONING_CODE", "ZONE_CODE"],
    "land_val": [
        "LND_VAL", "LAND_VAL", "LAND_VALUE", "LANDVAL", "JV_LAND",
    ],
    "bldg_val": [
        "IMP_VAL", "BLDG_VAL", "BUILDING_VAL", "BLDGVAL", "IMPRVAL",
        "IMPROVEMENT_VALUE", "IMPROV_VAL",
    ],
    "total_val": [
        "JV", "JUST_VAL", "JUSTVAL", "TOTAL_VAL", "MARKET_VAL",
        "TOTAL_ASSESSED_VALUE", "TV_NSD", "TOT_VAL", "TOT_MKT_VAL",
        "TOTAL_MARKET_VALUE",
    ],
    "lot_sf": ["LND_SQFOOT", "LAND_SQFT", "LOT_SF", "LOTSQFT", "SQFT"],
    "lot_ac": [
        "GIS_ACRES", "ACRES", "LAND_ACRES", "LOT_ACRES", "ACREAGE",
        "CALC_ACRES",
    ],
    "bldg_sf": [
        "TOT_LVG_AR", "TOT_LVG_AREA", "BLDG_SF", "BUILDING_SF",
        "HEATED_AREA", "GROSS_AREA", "TOTAL_LIVING_AREA", "BLDG_AREA",
    ],
    "year_built": [
        "ACT_YR_BLT", "EFF_YR_BLT", "YR_BLT", "YEAR_BUILT",
        "YEARBUILT", "YRBLT", "ACT_YEAR_BLT",
    ],
    "sale_date": [
        "SALE_DATE", "LAST_SALE_DATE", "LST_SALE_DT", "SALEDATE",
        "LSD",
    ],
    "sale_price": [
        "SALE_PRC1", "SALE_PRICE", "LAST_SALE_PRICE", "SALEPRICE",
        "LSP", "SALE_AMT",
    ],
    # Special-cased: some services split sale date into YR + MO columns.
    # The resolver additionally checks for SALE_YR1 / SALE_MO1 and
    # synthesizes sale_date from them if a direct field is not found.
}


@dataclass
class FieldMap:
    """Resolved canonical -> actual field name mapping for one layer."""
    layer_url: str
    mapping: dict[str, str]
    unresolved: list[str]

    def get(self, canonical: str) -> str | None:
        return self.mapping.get(canonical)

    def missing(self, required: list[str]) -> list[str]:
        return [c for c in required if c not in self.mapping]


def resolve_from_metadata(
    layer_url: str, layer_meta: dict
) -> FieldMap:
    """Given ?f=json output for a layer, resolve canonical -> actual names."""
    fields = layer_meta.get("fields") or []
    actual_names = {f["name"]: f["name"] for f in fields if "name" in f}
    lower_lookup = {name.lower(): name for name in actual_names}
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for canonical, candidates in ALIASES.items():
        found = None
        for cand in candidates:
            if cand.lower() in lower_lookup:
                found = lower_lookup[cand.lower()]
                break
        if found:
            mapping[canonical] = found
        else:
            unresolved.append(canonical)
    # Sale-date fallback: SALE_YR1 + SALE_MO1
    if "sale_date" in unresolved:
        yr = lower_lookup.get("sale_yr1") or lower_lookup.get("sale_year")
        mo = lower_lookup.get("sale_mo1") or lower_lookup.get("sale_month")
        if yr and mo:
            mapping["sale_date"] = f"__COMPOSITE__:{yr},{mo}"
            unresolved.remove("sale_date")
    return FieldMap(layer_url=layer_url, mapping=mapping, unresolved=unresolved)


REQUIRED_FOR_SCORING = [
    "parcel_id", "dor_uc", "land_val", "total_val",
    "lot_ac", "bldg_sf",
]
