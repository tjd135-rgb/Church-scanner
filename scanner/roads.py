"""Fetch US-1 and Dixie Highway centerlines and build a buffered corridor.

Uses osmnx to pull road linework from OSM, filtered to the tri-city
bounding box. Results are cached to disk so re-runs are cheap and
offline-friendly.

The buffered corridor is returned as a single unioned polygon in EPSG:3857
(meters), the CRS the extractor also uses for spatial joins.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

log = logging.getLogger(__name__)

# Bounding box covering Boca Raton -> Deerfield Beach -> Pompano Beach.
# west, south, east, north (lon/lat).
CORRIDOR_BBOX = (-80.20, 26.20, -80.05, 26.42)

FEET_PER_METER = 3.28084


def _bbox_to_polygon(bbox):
    from shapely.geometry import box
    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def load_road_centerlines(
    roads_config: list[dict], cache_dir: Path, bbox=CORRIDOR_BBOX
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame (EPSG:4326) of matching road edges."""
    cache_path = Path(cache_dir) / "road_centerlines.geojson"
    if cache_path.exists():
        try:
            gdf = gpd.read_file(cache_path)
            if len(gdf) > 0:
                log.info("loaded %d road edges from cache", len(gdf))
                return gdf
        except Exception as e:  # noqa: BLE001
            log.warning("cached centerlines unreadable (%s); re-fetching", e)

    import osmnx as ox

    log.info("fetching OSM road graph for corridor bbox %s", bbox)
    # `custom_filter` limits to highways likely to include US-1 / Dixie.
    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path(cache_dir) / "osm")
    Path(ox.settings.cache_folder).mkdir(parents=True, exist_ok=True)

    poly = _bbox_to_polygon(bbox)
    try:
        graph = ox.graph_from_polygon(
            poly,
            custom_filter=(
                '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            ),
            simplify=True,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"failed to fetch OSM road network: {e}. Check network access "
            f"or use a cached run."
        ) from e

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges = edges.reset_index()

    def _matches(row) -> str | None:
        name = row.get("name") or ""
        ref = row.get("ref") or ""
        if isinstance(name, list):
            name = " | ".join(str(x) for x in name)
        if isinstance(ref, list):
            ref = " | ".join(str(x) for x in ref)
        name = str(name)
        ref = str(ref)
        for r in roads_config:
            for pat in r.get("osm_ref_patterns", []) or []:
                if re.search(pat, ref, re.IGNORECASE):
                    return r["label"]
            for pat in r.get("osm_name_patterns", []) or []:
                if re.search(pat, name, re.IGNORECASE):
                    return r["label"]
        return None

    edges["corridor"] = edges.apply(_matches, axis=1)
    matched = edges[edges["corridor"].notna()].copy()
    if matched.empty:
        raise RuntimeError(
            "no OSM edges matched configured road patterns — check "
            "roads[].osm_ref_patterns / osm_name_patterns in config.yaml"
        )

    keep_cols = ["u", "v", "key", "name", "ref", "highway", "corridor",
                 "geometry"]
    matched = matched[[c for c in keep_cols if c in matched.columns]]
    matched = matched.set_crs("EPSG:4326") if matched.crs is None else matched
    matched.to_file(cache_path, driver="GeoJSON")
    log.info("cached %d corridor road edges -> %s", len(matched), cache_path)
    return matched


def build_corridor_buffer(
    edges: gpd.GeoDataFrame, buffer_ft: float
) -> gpd.GeoSeries:
    """Return a single dissolved MultiPolygon buffer (EPSG:3857, meters)."""
    if edges.empty:
        raise ValueError("no road edges to buffer")
    edges_m = edges.to_crs("EPSG:3857")
    buffer_m = buffer_ft / FEET_PER_METER
    buffered = edges_m.geometry.buffer(buffer_m)
    dissolved = unary_union(buffered.values)
    return gpd.GeoSeries([dissolved], crs="EPSG:3857")


def load_city_boundaries(
    client, cities_cfg: list[dict], source_layers: list[str]
) -> gpd.GeoDataFrame:
    """Fetch municipal polygons for the target cities.

    Returns EPSG:4326. The layer is resolved on first successful hit; the
    query filters by city name using OR clauses.
    """
    from .sources import Layer  # local import to avoid cycle at import time

    layer = client.resolve_layer(source_layers)
    fields_meta = {f["name"] for f in layer.metadata.get("fields", [])}
    name_field = None
    for cand in ("NAME", "CITY", "CITY_NAME", "MUNICIPALITY", "MUNI_NAME",
                 "PLACE_NAME", "NAMELSAD"):
        if cand in fields_meta:
            name_field = cand
            break
    if name_field is None:
        raise RuntimeError(
            f"city boundary layer {layer.url} has no obvious NAME field "
            f"(candidates: NAME/CITY/MUNI_NAME/etc.)"
        )
    city_names = [c["name"] for c in cities_cfg]
    where = " OR ".join(
        f"UPPER({name_field}) LIKE '%{n.upper()}%'" for n in city_names
    )
    features = client.query_all(
        layer, where=where, out_fields=[name_field], return_geometry=True,
    )
    if not features:
        raise RuntimeError(
            f"no city polygons returned. where={where}. field={name_field}"
        )
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry")
        if geom is None:
            continue
        rows.append({"city_name": props.get(name_field), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf
