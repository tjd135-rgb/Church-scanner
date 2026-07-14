"""Parcel extraction across three complementary passes.

Pass A — Land-use codes: query parcels whose DOR_UC is 71 or 72, clipped to
the city boundary polygons.

Pass B — Spatial buffer: query parcels intersecting the buffered road
corridor, then filter locally to DOR use codes 71/72 (some services only
allow one geometry filter at a time, so we always filter server-side by
geometry and client-side by use code to keep queries cheap).

Pass C — Situs-address regex: query parcels whose situs address string
matches one of the highway patterns AND whose DOR_UC is 71/72 — catches
parcels the buffer missed due to centerline imprecision.

Results from all three passes are unioned by parcel_id, with a `passes`
column marking which passes hit each row.
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

    Quotes the values if the layer's metadata reports the resolved field
    as an Esri string type; otherwise emits bare numeric literals. Never
    mixes quoted and unquoted values — that combination gets a 400 from
    some ArcGIS backends.
    """
    field = layer.field_map.get(canonical)
    if field is None:
        raise RuntimeError(
            f"cannot build IN() for canonical={canonical!r}: not mapped"
        )
    if layer.is_string_field(canonical):
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


def _fill_lot_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill lot_ac from lot_sf (or geometry) and vice-versa where missing."""
    if "lot_sf" in gdf.columns:
        gdf["lot_sf"] = pd.to_numeric(gdf["lot_sf"], errors="coerce")
    else:
        gdf["lot_sf"] = pd.NA
    if "lot_ac" in gdf.columns:
        gdf["lot_ac"] = pd.to_numeric(gdf["lot_ac"], errors="coerce")
    else:
        gdf["lot_ac"] = pd.NA

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


def extract_by_land_use(
    client: ArcGISClient,
    layer: Layer,
    city_polygons_4326: gpd.GeoDataFrame,
    dor_codes: list[int],
) -> gpd.GeoDataFrame:
    fm = layer.field_map
    dor_field = fm.get("dor_uc")
    if dor_field is None:
        raise RuntimeError(
            f"layer {layer.url} exposes no DOR use-code field. Cannot filter "
            f"by land use. field_map.unresolved={fm.unresolved}"
        )
    where = _in_clause(layer, "dor_uc", dor_codes)

    # Union city polygons and hand them to the client. If the union is an
    # axis-aligned rectangle (e.g. the corridor bbox fallback), the helper
    # will emit an envelope instead of a polygon — safer, no winding.
    esri_geom, geom_type = _cities_to_esri_geometry(city_polygons_4326)

    log.info("pass A (land use): querying %s where %s", layer.url, where)
    feats = client.query_all(
        layer,
        where=where,
        geometry=esri_geom, geometry_type=geom_type,
    )
    log.info("pass A returned %d features", len(feats))
    gdf = _canonicalize(feats, fm)
    gdf["pass_land_use"] = True
    return gdf


def extract_by_buffer(
    client: ArcGISClient,
    layer: Layer,
    corridor_buffer_3857: gpd.GeoSeries,
    dor_codes: list[int] | None,
) -> gpd.GeoDataFrame:
    fm = layer.field_map
    corridor_4326 = corridor_buffer_3857.to_crs("EPSG:4326")
    esri_geom, geom_type = shapely_to_esri_geometry(corridor_4326.geometry.iloc[0])
    where = "1=1"
    if dor_codes:
        dor_field = fm.get("dor_uc")
        if dor_field:
            where = _in_clause(layer, "dor_uc", dor_codes)
    log.info("pass B (buffer): querying %s where %s", layer.url, where)
    feats = client.query_all(
        layer, where=where, geometry=esri_geom,
        geometry_type=geom_type,
    )
    log.info("pass B returned %d features", len(feats))
    gdf = _canonicalize(feats, fm)
    gdf["pass_buffer"] = True
    return gdf


def extract_by_address(
    client: ArcGISClient,
    layer: Layer,
    city_polygons_4326: gpd.GeoDataFrame,
    address_patterns: list[str],
    dor_codes: list[int],
) -> gpd.GeoDataFrame:
    fm = layer.field_map
    situs_field = fm.get("situs_addr")
    dor_field = fm.get("dor_uc")
    if situs_field is None or dor_field is None:
        log.warning(
            "pass C skipped: situs_addr=%s dor_uc=%s",
            situs_field, dor_field,
        )
        return gpd.GeoDataFrame(columns=list(fm.mapping.keys()) + ["geometry"],
                                geometry="geometry", crs="EPSG:4326")

    like_clauses = " OR ".join(
        f"UPPER({situs_field}) LIKE '%{p}%'" for p in address_patterns
    )
    codes_clause = _in_clause(layer, "dor_uc", dor_codes)
    where = f"({like_clauses}) AND {codes_clause}"

    esri_geom, geom_type = _cities_to_esri_geometry(city_polygons_4326)
    log.info("pass C (address): querying %s where %s", layer.url, where)
    feats = client.query_all(
        layer, where=where, geometry=esri_geom,
        geometry_type=geom_type,
    )
    log.info("pass C returned %d features", len(feats))
    gdf = _canonicalize(feats, fm)
    gdf["pass_address"] = True
    return gdf


def merge_passes(*gdfs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    non_empty = [g for g in gdfs if g is not None and not g.empty]
    if not non_empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    all_cols = set().union(*(g.columns for g in non_empty))
    for g in non_empty:
        for c in ("pass_land_use", "pass_buffer", "pass_address"):
            if c not in g.columns:
                g[c] = False
        for c in all_cols - set(g.columns):
            g[c] = pd.NA
    merged = pd.concat(non_empty, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry",
                              crs=non_empty[0].crs)

    def _agg(group: pd.DataFrame) -> pd.Series:
        pick = group.iloc[0].copy()
        pick["pass_land_use"] = bool(group["pass_land_use"].any())
        pick["pass_buffer"] = bool(group["pass_buffer"].any())
        pick["pass_address"] = bool(group["pass_address"].any())
        return pick

    if "parcel_id" not in merged.columns:
        return merged
    merged["parcel_id"] = merged["parcel_id"].astype("string").fillna("")
    keyed = merged[merged["parcel_id"] != ""]
    keyless = merged[merged["parcel_id"] == ""]
    if len(keyed):
        keyed = keyed.groupby("parcel_id", as_index=False, sort=False).apply(_agg)
        keyed = keyed.reset_index(drop=True)
    combined = pd.concat([keyed, keyless], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry",
                                crs=merged.crs)
    combined = _numeric_columns(combined)
    combined = _fill_lot_metrics(combined)
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


def _shapely_to_esri_envelope(geom) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    return {
        "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
        "spatialReference": {"wkid": 4326},
    }


def _orient_cw(polygon):
    """Return the polygon with exterior CW and interior rings CCW."""
    from shapely.geometry.polygon import orient
    # sign=-1.0 -> exterior CW, interiors CCW (Esri convention).
    return orient(polygon, sign=-1.0)


def _shapely_to_esri_polygon(geom) -> dict:
    """Convert a (Multi)Polygon in EPSG:4326 to an Esri JSON polygon with
    Esri-compliant ring winding (exterior CW, interior CCW)."""
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
    return {"rings": rings, "spatialReference": {"wkid": 4326}}


def shapely_to_esri_geometry(geom) -> tuple[dict, str]:
    """Return (esri_geom_json, esriGeometryType) for a shapely geometry.

    Prefers an envelope when the input is an axis-aligned rectangle since
    envelopes carry no winding ambiguity."""
    if _is_axis_aligned_rectangle(geom):
        return _shapely_to_esri_envelope(geom), "esriGeometryEnvelope"
    return _shapely_to_esri_polygon(geom), "esriGeometryPolygon"


def _cities_to_esri_geometry(
    city_polygons_4326: gpd.GeoDataFrame,
) -> tuple[dict, str]:
    from shapely.ops import unary_union
    u = unary_union(city_polygons_4326.geometry.values)
    # Simplify slightly so the query payload stays under service limits.
    u = u.simplify(0.0005, preserve_topology=True)
    return shapely_to_esri_geometry(u)
