"""Parcel extraction: one bounded server-side query + three client-side passes.

The FGIO Statewide Cadastral service rejects any query that combines a
WHERE clause with a geometry filter — verified in --probe-query L3d/L3e
(200-err even with correctly-projected native-SR envelope). So we can't
push both filters server-side. Strategy:

1. **Single server-side query**: DOR_UC IN (71,72) AND CO_NO IN (<target
   counties>). Bounded by county so we don't paginate all-of-Florida
   just to get 30k rows.
2. **Three client-side passes** applied as flags on the returned frame:
   - pass_land_use: parcel falls within any target city polygon (or the
     corridor bbox fallback).
   - pass_buffer: parcel intersects the buffered road corridor.
   - pass_address: situs address string matches one of the highway
     regex patterns.
3. Keep any row with pass_land_use OR pass_buffer OR pass_address True.

This is a superset of the original spec's "either method" union and
handles the service's constraint cleanly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping

from . import fields as F
from .sources import ArcGISClient, Layer

log = logging.getLogger(__name__)

# Meters^2 per acre (used to derive lot_ac from lot_sf when only one is present).
SQFT_PER_ACRE = 43560.0
SQM_PER_ACRE = 4046.8564224


@dataclass
class ExtractionResult:
    parcels: gpd.GeoDataFrame  # canonical columns + geometry (EPSG:4326)
    layer_used: str


def _in_clause(layer: Layer, canonical: str, values: list) -> str:
    """Build a syntactically clean `field IN (...)` clause.

    Quotes values based on Esri field type from cached metadata; never
    mixes quoted and unquoted (some ArcGIS backends 400 on that).

    Special case for `dor_uc`: FGIO Statewide Cadastral reports DOR_UC
    as esriFieldTypeString but rejects quoted-literal comparisons with
    'Cannot perform query. Invalid query parameters.' — verified by the
    --probe-query L1a/L1b test. DOR use codes are always integers 0-99,
    so we compare unquoted regardless of the metadata's reported type.
    """
    field = layer.field_map.get(canonical)
    if field is None:
        raise RuntimeError(
            f"cannot build IN() for canonical={canonical!r}: not mapped"
        )
    if canonical == "dor_uc":
        parts = ",".join(str(int(v)) for v in values)
    elif layer.is_string_field(canonical):
        parts = ",".join(f"'{str(v)}'" for v in values)
    else:
        parts = ",".join(str(int(v)) for v in values)
    return f"{field} IN ({parts})"


def _canonicalize(
    features: list[dict], fm: F.FieldMap
) -> gpd.GeoDataFrame:
    rows = []
    for feat in features:
        props = feat.get("properties", {}) or feat.get("attributes", {}) or {}
        # Uppercase-normalize key lookup — services occasionally serve
        # lowercased attribute keys on GeoJSON responses.
        props_ci = {k.lower(): v for k, v in props.items()}
        row: dict = {}
        for canonical, actual in fm.mapping.items():
            if actual.startswith("__COMPOSITE__:"):
                yr_field, mo_field = actual.split(":", 1)[1].split(",")
                yr = props_ci.get(yr_field.lower())
                mo = props_ci.get(mo_field.lower())
                if yr and mo:
                    try:
                        row[canonical] = datetime(int(yr), max(1, int(mo)), 1)
                    except Exception:  # noqa: BLE001
                        row[canonical] = None
                else:
                    row[canonical] = None
            else:
                row[canonical] = props_ci.get(actual.lower())
        geom = feat.get("geometry")
        row["geometry"] = shape(geom) if geom else None
        rows.append(row)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def _fill_derived_bldg_val(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute bldg_val = total_val - land_val where the layer doesn't
    expose an improvement value directly. FGDL statewide is the driver:
    it carries JV (total) and LND_VAL but no IMP_VAL."""
    if "bldg_val" not in gdf.columns:
        gdf["bldg_val"] = pd.NA
    gdf["bldg_val"] = pd.to_numeric(gdf["bldg_val"], errors="coerce")
    tv = pd.to_numeric(gdf.get("total_val"), errors="coerce")
    lv = pd.to_numeric(gdf.get("land_val"), errors="coerce")
    mask = gdf["bldg_val"].isna() & tv.notna() & lv.notna()
    if mask.any():
        derived = (tv - lv).clip(lower=0)
        gdf.loc[mask, "bldg_val"] = derived[mask]
    return gdf


def _fill_lot_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill lot_ac from lot_sf (or geometry) and vice-versa where missing."""
    # Ensure both columns exist as float64 (not object/NA) so later
    # .loc[mask] = <float> assignments don't upcast on write.
    nan_col = pd.Series([float("nan")] * len(gdf), index=gdf.index, dtype="float64")
    if "lot_sf" in gdf.columns:
        gdf["lot_sf"] = pd.to_numeric(gdf["lot_sf"], errors="coerce")
    else:
        gdf["lot_sf"] = nan_col.copy()
    if "lot_ac" in gdf.columns:
        gdf["lot_ac"] = pd.to_numeric(gdf["lot_ac"], errors="coerce")
    else:
        gdf["lot_ac"] = nan_col.copy()

    # Fill acres from square feet where possible.
    mask = gdf["lot_ac"].isna() & gdf["lot_sf"].notna()
    gdf.loc[mask, "lot_ac"] = gdf.loc[mask, "lot_sf"] / SQFT_PER_ACRE
    # Fill sqft from acres where possible.
    mask = gdf["lot_sf"].isna() & gdf["lot_ac"].notna()
    gdf.loc[mask, "lot_sf"] = gdf.loc[mask, "lot_ac"] * SQFT_PER_ACRE

    # As a last resort, derive from geometry area (EPSG:3857 approximation
    # for meters — accurate enough for a redevelopment screen).
    needs_geom = gdf["lot_ac"].isna() & gdf.geometry.notna()
    if needs_geom.any():
        proj = gdf.loc[needs_geom].to_crs("EPSG:3857")
        acres = proj.geometry.area / SQM_PER_ACRE
        gdf.loc[needs_geom, "lot_ac"] = acres.values
        gdf.loc[needs_geom & gdf["lot_sf"].isna(), "lot_sf"] = (
            acres.values * SQFT_PER_ACRE
        )
    return gdf


def _numeric_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for c in ("land_val", "bldg_val", "total_val", "bldg_sf", "sale_price",
              "year_built", "dor_uc"):
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
    if "sale_date" in gdf.columns:
        gdf["sale_date"] = pd.to_datetime(gdf["sale_date"], errors="coerce")
    return gdf


def query_target_parcels(
    client: ArcGISClient,
    layer: Layer,
    dor_codes: list[int],
    county_numbers: list[int],
) -> gpd.GeoDataFrame:
    """Server-side WHERE-only query: DOR_UC IN (…) AND CO_NO IN (…).

    Returns geometry in EPSG:4326 (client asks for outSR=4326). County
    bounding keeps the payload manageable — statewide DOR 71/72 would
    be ~30-50k rows; two counties trim that to a few thousand.

    Requests only the ~20 fields our field_map actually reads (plus
    OBJECTID for stable pagination) rather than outFields=*. FGIO's
    Statewide Cadastral has ~118 columns; asking for all of them
    combined with returnGeometry=true pushes the query past whatever
    internal cost limit produces 'Unable to perform query.'
    """
    fm = layer.field_map
    dor_field = fm.get("dor_uc")
    if dor_field is None:
        raise RuntimeError(
            f"layer {layer.url} exposes no DOR use-code field. "
            f"unresolved={fm.unresolved}"
        )
    where = _in_clause(layer, "dor_uc", dor_codes)
    if county_numbers and fm.get("county"):
        # CO_NO on the FGIO layer is esriFieldTypeDouble; _in_clause
        # emits unquoted numeric literals for non-string fields.
        where += " AND " + _in_clause(layer, "county", county_numbers)

    out_fields = _fields_needed_for_extraction(layer)
    log.info(
        "querying %s where %s (outFields=%d cols)",
        layer.url, where, len(out_fields),
    )
    feats = client.query_all(layer, where=where, out_fields=out_fields)
    log.info("server returned %d features", len(feats))
    gdf = _canonicalize(feats, fm)
    return gdf


def _fields_needed_for_extraction(layer: Layer) -> list[str]:
    """Return the actual layer field names we need to read (canonical
    field_map values + OBJECTID for stable pagination). Expands the
    special __COMPOSITE__ marker into its component fields."""
    fm = layer.field_map
    needed: set[str] = set()
    oid = layer.metadata.get("objectIdField") or "OBJECTID"
    needed.add(oid)
    for actual in fm.mapping.values():
        if actual.startswith("__COMPOSITE__:"):
            parts = actual.split(":", 1)[1].split(",")
            needed.update(p.strip() for p in parts if p.strip())
        else:
            needed.add(actual)
    return sorted(needed)


def apply_pass_flags(
    gdf: gpd.GeoDataFrame,
    city_polygons_4326: gpd.GeoDataFrame,
    corridor_buffer_3857: gpd.GeoSeries,
    address_patterns: list[str],
) -> gpd.GeoDataFrame:
    """Add pass_land_use / pass_buffer / pass_address flags client-side.

    - pass_land_use: parcel intersects any city polygon
    - pass_buffer:   parcel intersects the buffered road corridor
    - pass_address:  situs address string matches an address pattern
    """
    import re

    if gdf.empty:
        for c in ("pass_land_use", "pass_buffer", "pass_address"):
            gdf[c] = False
        return gdf

    proj_parcels = gdf.to_crs("EPSG:3857")

    # pass_land_use
    if city_polygons_4326 is not None and not city_polygons_4326.empty:
        from shapely.ops import unary_union
        city_union_3857 = unary_union(
            city_polygons_4326.to_crs("EPSG:3857").geometry.values
        )
        gdf["pass_land_use"] = proj_parcels.geometry.intersects(city_union_3857).values
    else:
        gdf["pass_land_use"] = True  # no city constraint

    # pass_buffer
    if corridor_buffer_3857 is not None and len(corridor_buffer_3857):
        buf = corridor_buffer_3857.iloc[0]
        gdf["pass_buffer"] = proj_parcels.geometry.intersects(buf).values
    else:
        gdf["pass_buffer"] = False

    # pass_address
    if "situs_addr" in gdf.columns and address_patterns:
        addrs = gdf["situs_addr"].astype("string").str.upper().fillna("")
        pattern = "|".join(re.escape(p) for p in address_patterns)
        gdf["pass_address"] = addrs.str.contains(pattern, regex=True, na=False).values
    else:
        gdf["pass_address"] = False

    log.info(
        "pass counts: land_use=%d, buffer=%d, address=%d",
        int(gdf["pass_land_use"].sum()),
        int(gdf["pass_buffer"].sum()),
        int(gdf["pass_address"].sum()),
    )
    return gdf


def filter_to_any_pass(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only rows caught by at least one pass."""
    if gdf.empty:
        return gdf
    mask = (
        gdf["pass_land_use"].fillna(False).astype(bool)
        | gdf["pass_buffer"].fillna(False).astype(bool)
        | gdf["pass_address"].fillna(False).astype(bool)
    )
    return gdf[mask].reset_index(drop=True)


def finalize_frame(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Numeric coercion + derived lot/bldg fields on the flagged frame."""
    if gdf.empty:
        return gdf
    combined = _numeric_columns(gdf)
    combined = _fill_lot_metrics(combined)
    combined = _fill_derived_bldg_val(combined)
    return combined


def spatial_flag_in_buffer(
    gdf: gpd.GeoDataFrame, corridor_buffer_3857: gpd.GeoSeries
) -> gpd.GeoDataFrame:
    """Add `in_corridor_buffer` column via a client-side spatial test."""
    if gdf.empty:
        gdf["in_corridor_buffer"] = pd.Series(dtype=bool)
        return gdf
    proj = gdf.to_crs("EPSG:3857")
    buf = corridor_buffer_3857.iloc[0]
    proj["in_corridor_buffer"] = proj.geometry.intersects(buf)
    gdf["in_corridor_buffer"] = proj["in_corridor_buffer"].values
    return gdf


# ---- Esri geometry helpers -------------------------------------------
#
# Esri REST API winding convention: exterior rings must be CLOCKWISE and
# interior (hole) rings must be COUNTER-CLOCKWISE. A CCW exterior ring is
# reinterpreted as a hole, which yields a malformed polygon and a 400
# "Unable to perform query. Please check your parameters." Shapely's
# default winding (from shapely.geometry.box, OSM data, shapefiles) is
# CCW exterior — the opposite of what Esri wants — so we must orient
# every polygon before emitting it. Also: when the input is a simple
# axis-aligned rectangle (e.g. the corridor bbox fallback), emit an
# esriGeometryEnvelope instead — envelopes have no winding at all.


def _is_axis_aligned_rectangle(geom, tol: float = 1e-9) -> bool:
    """True iff geom is a Polygon whose exterior is a 4-corner rectangle
    aligned to the x/y axes and with no interior rings. tol handles the
    tiny numeric jitter introduced by shapely operations."""
    if geom.geom_type != "Polygon" or list(geom.interiors):
        return False
    coords = list(geom.exterior.coords)
    if len(coords) not in (4, 5):
        return False
    minx, miny, maxx, maxy = geom.bounds
    expected = {(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)}
    actual = set()
    for x, y in coords[:-1] if len(coords) == 5 else coords:
        actual.add((round(x, 12), round(y, 12)))
    expected_r = {(round(x, 12), round(y, 12)) for x, y in expected}
    return actual == expected_r


def native_wkid(layer: Layer) -> int:
    """Return the layer's native spatial-reference WKID (from extent).

    Some enterprise ArcGIS services can't reproject an input geometry
    from a client SR (like 4326) into the layer's native SR on the fly
    — they return 'Cannot perform query. Invalid query parameters.' for
    any spatial filter. Sending the geometry pre-projected to the
    layer's own SR avoids the round-trip. Verified in --probe-query
    L3d against Florida_Statewide_Cadastral (native SR 3086)."""
    ext = layer.metadata.get("extent") or {}
    sr = ext.get("spatialReference") or {}
    return int(sr.get("latestWkid") or sr.get("wkid") or 4326)


def _project_geom(geom, source_wkid: int, target_wkid: int):
    """Reproject a shapely geometry between two WKIDs (skips no-op)."""
    if source_wkid == target_wkid:
        return geom
    from pyproj import Transformer
    from shapely.ops import transform
    tf = Transformer.from_crs(source_wkid, target_wkid, always_xy=True)
    return transform(tf.transform, geom)


def _shapely_to_esri_envelope(geom, wkid: int = 4326) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    return {
        "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
        "spatialReference": {"wkid": wkid},
    }


def _orient_cw(polygon):
    """Return the polygon with exterior CW and interior rings CCW."""
    from shapely.geometry.polygon import orient
    # sign=-1.0 -> exterior CW, interiors CCW (Esri convention).
    return orient(polygon, sign=-1.0)


def _shapely_to_esri_polygon(geom, wkid: int = 4326) -> dict:
    """Convert a (Multi)Polygon to an Esri JSON polygon with Esri-compliant
    ring winding (exterior CW, interior CCW). Caller is responsible for
    supplying the geometry already in the target SR."""
    if geom.geom_type == "Polygon":
        polys = [_orient_cw(geom)]
    elif geom.geom_type == "MultiPolygon":
        polys = [_orient_cw(p) for p in geom.geoms]
    else:
        raise ValueError(f"expected polygon geometry, got {geom.geom_type}")
    rings: list[list[list[float]]] = []
    for p in polys:
        rings.append([[x, y] for x, y in p.exterior.coords])
        for interior in p.interiors:
            rings.append([[x, y] for x, y in interior.coords])
    return {"rings": rings, "spatialReference": {"wkid": wkid}}


def shapely_to_esri_geometry(
    geom_4326, target_wkid: int = 4326,
) -> tuple[dict, str]:
    """Return (esri_geom_json, esriGeometryType) for a shapely geometry.

    The input is expected in EPSG:4326. If `target_wkid` differs, the
    geometry is reprojected client-side before serialization — that
    avoids relying on the ArcGIS server to reproject the filter, which
    some large enterprise services can't or won't do.

    Envelope preference is decided on the INPUT geometry (before
    projection): if the caller handed us an axis-aligned rectangle in
    4326, we emit the projected geometry's bounding box as an envelope
    even though projection may have introduced tiny corner distortion.
    That's a superset filter — downstream client-side filtering
    catches any extra parcels near the corners — and gives us the
    envelope-simplicity payoff without a winding gotcha."""
    was_bbox = _is_axis_aligned_rectangle(geom_4326)
    projected = _project_geom(geom_4326, 4326, target_wkid)
    if was_bbox:
        return (
            _shapely_to_esri_envelope(projected, wkid=target_wkid),
            "esriGeometryEnvelope",
        )
    return (
        _shapely_to_esri_polygon(projected, wkid=target_wkid),
        "esriGeometryPolygon",
    )


def _cities_to_esri_geometry(
    city_polygons_4326: gpd.GeoDataFrame, target_wkid: int = 4326,
) -> tuple[dict, str]:
    from shapely.ops import unary_union
    u = unary_union(city_polygons_4326.geometry.values)
    # Simplify slightly (in 4326 degrees) so the query payload stays under
    # service limits; the projection below is unaffected by the simplify.
    u = u.simplify(0.0005, preserve_topology=True)
    return shapely_to_esri_geometry(u, target_wkid=target_wkid)
