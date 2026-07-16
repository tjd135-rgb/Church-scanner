"""End-to-end orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import export as export_mod
from . import extract as extract_mod
from . import metrics as metrics_mod
from . import roads as roads_mod
from . import validate as validate_mod
from .config import Config
from .sources import ArcGISClient, ServiceError

log = logging.getLogger(__name__)


def run(
    cfg: Config,
    output_dir: Path,
    buffer_ft: float | None = None,
    skip_validation: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    buf_ft = buffer_ft if buffer_ft is not None else cfg.buffer_ft
    cache_dir = cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = ArcGISClient(
        cache_dir=cache_dir,
        timeout_s=cfg.runtime.get("request_timeout_s", 60),
        max_retries=cfg.runtime.get("max_retries", 4),
        page_size=cfg.runtime.get("page_size", 2000),
    )

    # 1. Resolve the parcel layer. Try statewide first; fall back per county.
    log.info("resolving parcel layer(s)")
    parcel_layer = None
    for key in ("florida_statewide_parcels", "broward_bcpa", "palm_beach_papa"):
        source = cfg.sources.get(key)
        if not source:
            continue
        try:
            parcel_layer = client.resolve_layer(source.get("layers", []))
            log.info("using parcel layer: %s (source key: %s)",
                     parcel_layer.url, key)
            break
        except ServiceError as e:
            log.warning("source '%s' unavailable: %s", key, e)
    if parcel_layer is None:
        raise RuntimeError(
            "no parcel layer reachable. Check config.yaml sources[] URLs "
            "and outbound network access to ArcGIS Feature/Map Services."
        )

    # 2. Load city boundaries. If the city layer can't be resolved, degrade
    # to a bounding-box polygon so the pipeline still completes.
    log.info("loading city boundaries")
    try:
        city_polys = roads_mod.load_city_boundaries(
            client, cfg.cities,
            cfg.sources.get("city_boundaries", {}).get("layers", []),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("city boundary layer unavailable (%s); "
                    "falling back to corridor bounding box", e)
        from shapely.geometry import box
        bbox = roads_mod.CORRIDOR_BBOX
        city_polys = gpd.GeoDataFrame(
            [{"city_name": "corridor_bbox",
              "geometry": box(bbox[0], bbox[1], bbox[2], bbox[3])}],
            geometry="geometry", crs="EPSG:4326",
        )

    # 3. Fetch road centerlines and build the buffer.
    log.info("fetching road centerlines")
    edges = roads_mod.load_road_centerlines(cfg.roads, cache_dir)
    corridor = roads_mod.build_corridor_buffer(edges, buf_ft)

    # 4. Server-side WHERE-only query, then three client-side passes.
    # This service (FGIO Statewide Cadastral) rejects combined WHERE +
    # geometry filters, so we can't push both server-side. Instead:
    # bound by county server-side, apply spatial + address filters
    # client-side. See scanner/extract.py docstring for details.
    county_numbers = sorted({
        int(c["fl_county_no"]) for c in cfg.cities
        if c.get("fl_county_no") is not None
    })
    log.info("running server-side query bounded by counties %s",
             county_numbers)
    fetched = extract_mod.query_target_parcels(
        client, parcel_layer, cfg.dor_use_codes, county_numbers,
    )
    if fetched.empty:
        raise RuntimeError(
            "server-side query returned no DOR 71/72 parcels — check "
            "county numbers in config.yaml"
        )
    log.info("applying client-side pass flags")
    flagged = extract_mod.apply_pass_flags(
        fetched, city_polys, corridor, cfg.address_patterns,
    )
    merged = extract_mod.filter_to_any_pass(flagged)
    merged = extract_mod.finalize_frame(merged)
    log.info("kept %d parcels caught by at least one pass", len(merged))

    if merged.empty:
        log.error("no parcels matched — halting before metrics/export")
        raise RuntimeError("no parcels matched any of the three passes")

    merged["source_layer"] = parcel_layer.url

    # 5. Flag proximity to buffer (used as a display field alongside
    # pass_buffer — both mean the same thing, kept for compatibility).
    merged = extract_mod.spatial_flag_in_buffer(merged, corridor)

    # 6. Metrics + composite score.
    merged = metrics_mod.compute_metrics(
        merged, cfg.score_weights, cfg.score_caps,
    )

    # 7. Validation.
    hits = validate_mod.validate(merged, cfg.validation_parcels)
    val_report = validate_mod.report(hits)
    log.info("\n%s", val_report)
    if not skip_validation:
        missed = [h for h in hits if not h.matched]
        if missed:
            log.error(
                "VALIDATION FAILURE — the pipeline missed known parcels. "
                "Re-run with --skip-validation to export anyway, or widen "
                "buffer_ft / address_patterns / source layer URLs."
            )
    # 8. Prepare output.
    prepared = export_mod.prepare_output(pd.DataFrame(merged.drop(columns="geometry", errors="ignore")))

    summary = export_mod.summary_lines(
        prepared,
        scoring_docs=metrics_mod.weight_documentation(cfg.score_weights, cfg.score_caps),
        buffer_ft=buf_ft,
        cities=[c["name"] for c in cfg.cities],
        layer_used=parcel_layer.url,
        validation_report=val_report,
    )

    csv_path = output_dir / "church_school_parcels.csv"
    xlsx_path = output_dir / "church_school_parcels.xlsx"
    export_mod.write_csv(prepared, csv_path)
    export_mod.write_xlsx(prepared, summary, xlsx_path)

    # Also save the full raw dataset (with geometry) as GeoJSON for GIS re-use.
    geojson_path = output_dir / "church_school_parcels.geojson"
    merged.to_file(geojson_path, driver="GeoJSON")
    log.info("wrote %s", geojson_path)

    return {
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
        "geojson": str(geojson_path),
        "count": len(prepared),
        "validation": [h.__dict__ for h in hits],
        "layer_used": parcel_layer.url,
    }
