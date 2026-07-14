#!/usr/bin/env python3
"""Single-command entry point.

Usage:
    python run.py                       # default config, ./output
    python run.py --buffer-ft 1000
    python run.py --output-dir /tmp/parcel-scan
    python run.py --skip-validation
    python run.py --config myconfig.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scanner import config as cfg_mod
from scanner import pipeline as pipe_mod


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # osmnx and urllib3 are chatty at DEBUG; keep them at INFO unless we're
    # really debugging.
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("osmnx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Scan Florida parcels for churches (DOR 71) and private schools "
            "(DOR 72) along US-1 / Dixie Highway from Boca Raton through "
            "Deerfield Beach to Pompano Beach."
        )
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config.yaml (default: bundled config.yaml)")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="Directory to write CSV/XLSX/GeoJSON into.")
    p.add_argument("--buffer-ft", type=float, default=None,
                   help="Override buffer_ft from config.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Do not fail loudly if known validation parcels miss.")
    p.add_argument("--dump-metadata", action="store_true",
                   help=(
                       "Do not run the pipeline. Resolve the parcel layer, "
                       "print every field and its Esri type, then exit. Use "
                       "this to diagnose 400s from unexpected field names."
                   ))
    p.add_argument("--probe-query", action="store_true",
                   help=(
                       "Do not run the pipeline. Resolve the parcel layer "
                       "and fire a sequence of minimal queries against its "
                       "/query endpoint, escalating one parameter at a time "
                       "(f=json vs geojson, outFields=* vs OBJECTID, "
                       "resultRecordCount, returnGeometry, geometry filter). "
                       "Report which combinations succeed and which 400. "
                       "Use to isolate the exact rejected parameter."
                   ))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("run")

    cfg = cfg_mod.load(args.config)
    log.info("loaded config from %s", cfg.path)

    if args.dump_metadata:
        return _dump_metadata(cfg)

    if args.probe_query:
        return _probe_query(cfg)

    result = pipe_mod.run(
        cfg,
        output_dir=args.output_dir.resolve(),
        buffer_ft=args.buffer_ft,
        skip_validation=args.skip_validation,
    )
    log.info("done. %d parcels. csv=%s xlsx=%s",
             result["count"], result["csv"], result["xlsx"])
    # Exit non-zero if validation flagged misses and we were told to enforce it.
    if not args.skip_validation:
        missed = [v for v in result["validation"] if not v["matched"]]
        if missed:
            log.error("%d validation parcels missing — see log above",
                      len(missed))
            return 2
    return 0


def _dump_metadata(cfg) -> int:
    """Resolve each configured parcel source and print its field list.

    This does no querying — only the ?f=json metadata fetch — so it works
    for diagnosing schema drift even when the query endpoint is misbehaving.
    """
    from scanner.sources import ArcGISClient, ServiceError

    client = ArcGISClient(
        cache_dir=cfg.cache_dir,
        timeout_s=cfg.runtime.get("request_timeout_s", 60),
        max_retries=cfg.runtime.get("max_retries", 4),
        page_size=cfg.runtime.get("page_size", 2000),
    )
    for key in ("florida_statewide_parcels", "broward_bcpa", "palm_beach_papa"):
        src = cfg.sources.get(key)
        if not src:
            continue
        print(f"\n=== {key} ===")
        try:
            layer = client.resolve_layer(src.get("layers", []))
        except ServiceError as e:
            print(f"  unavailable: {e}")
            continue
        print(f"  url: {layer.url}")
        print(f"  maxRecordCount: {layer.max_record_count}")
        print(f"  supportsPagination: {layer.supports_pagination}")
        print(f"  fields:")
        for f in layer.metadata.get("fields", []) or []:
            print(f"    {f.get('name'):<32} {f.get('type'):<24} "
                  f"alias={f.get('alias')}")
        print(f"  canonical field map:")
        for c, actual in layer.field_map.mapping.items():
            t = layer.field_type(c) or "n/a"
            print(f"    {c:<20} -> {actual:<24} ({t})")
        if layer.field_map.unresolved:
            print(f"  unresolved canonical fields: "
                  f"{layer.field_map.unresolved}")
    return 0


def _probe_query(cfg) -> int:
    """Isolate the parameter that makes the /query endpoint return 400.

    Fires a graded series of GETs against the parcel layer's /query
    endpoint. Each row is one variable changed relative to the previous
    successful baseline; when a row 400s, the parameter added in that
    row is the trigger.
    """
    import json
    import requests
    from scanner.sources import ArcGISClient, ServiceError

    client = ArcGISClient(
        cache_dir=cfg.cache_dir,
        timeout_s=cfg.runtime.get("request_timeout_s", 60),
        max_retries=0,  # no retry noise; a 400 is a 400
        page_size=cfg.runtime.get("page_size", 2000),
    )
    src = cfg.sources.get("florida_statewide_parcels")
    if not src:
        print("no florida_statewide_parcels source configured")
        return 1
    try:
        layer = client.resolve_layer(src.get("layers", []))
    except ServiceError as e:
        print(f"could not resolve any candidate layer: {e}")
        return 1
    print(f"probing layer: {layer.url}")
    print(f"  supportedQueryFormats = "
          f"{layer.metadata.get('supportedQueryFormats')!r}")
    print(f"  maxRecordCount        = {layer.max_record_count}")
    print(f"  supportsPagination    = {layer.supports_pagination}")
    dor_field = layer.field_map.get("dor_uc")
    if dor_field is None:
        print("  no DOR use-code field mapped — cannot build WHERE. abort.")
        return 1
    where = (
        f"{dor_field} IN ('71','72')"
        if layer.is_string_field("dor_uc")
        else f"{dor_field} IN (71,72)"
    )
    pid_field = layer.field_map.get("parcel_id") or "OBJECTID"

    probes: list[tuple[str, dict]] = [
        # (label, params) — each row adds one thing to the previous.
        ("baseline: f=json, where only, OBJECTID, count=1, no geom",
         {"f": "json", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 1,
          "returnGeometry": "false"}),
        ("+ outFields=*",
         {"f": "json", "where": where,
          "outFields": "*", "resultRecordCount": 1,
          "returnGeometry": "false"}),
        ("+ outFields=OBJECTID," + str(pid_field),
         {"f": "json", "where": where,
          "outFields": f"OBJECTID,{pid_field}",
          "resultRecordCount": 1, "returnGeometry": "false"}),
        ("+ resultRecordCount=100",
         {"f": "json", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 100,
          "returnGeometry": "false"}),
        ("+ returnGeometry=true (f=json)",
         {"f": "json", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 1,
          "returnGeometry": "true", "outSR": 4326}),
        ("+ f=geojson (no geometry filter, returnGeometry=true)",
         {"f": "geojson", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 1,
          "returnGeometry": "true", "outSR": 4326}),
        ("+ envelope filter (f=json, envelope covering tri-city area)",
         {"f": "json", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 1,
          "returnGeometry": "false",
          "geometry": json.dumps({
              "xmin": -80.20, "ymin": 26.20,
              "xmax": -80.05, "ymax": 26.42,
              "spatialReference": {"wkid": 4326},
          }),
          "geometryType": "esriGeometryEnvelope",
          "spatialRel": "esriSpatialRelIntersects",
          "inSR": 4326}),
        ("+ envelope filter with f=geojson",
         {"f": "geojson", "where": where,
          "outFields": "OBJECTID", "resultRecordCount": 1,
          "returnGeometry": "true", "outSR": 4326,
          "geometry": json.dumps({
              "xmin": -80.20, "ymin": 26.20,
              "xmax": -80.05, "ymax": 26.42,
              "spatialReference": {"wkid": 4326},
          }),
          "geometryType": "esriGeometryEnvelope",
          "spatialRel": "esriSpatialRelIntersects",
          "inSR": 4326}),
    ]

    endpoint = f"{layer.url}/query"
    session = requests.Session()
    session.headers.update({"User-Agent": "church-scanner-probe/1.0"})
    print(f"\nendpoint = {endpoint}")
    print(f"WHERE    = {where}\n")
    print(f"{'#':>2}  {'status':>6}  {'nfeat':>5}  label")
    print(f"{'-'*2}  {'-'*6}  {'-'*5}  {'-'*60}")

    for i, (label, params) in enumerate(probes, 1):
        try:
            r = session.get(endpoint, params=params, timeout=60)
            status = r.status_code
            note = ""
            try:
                body = r.json()
            except ValueError:
                body = None
                note = "non-JSON response"
            if isinstance(body, dict) and "error" in body:
                err = body["error"]
                note = (
                    f"error code={err.get('code')} "
                    f"details={err.get('details')}"
                )
                nfeat = "-"
                status = f"{status} err"
            elif isinstance(body, dict):
                nfeat = len(body.get("features") or [])
            else:
                nfeat = "-"
        except Exception as e:  # noqa: BLE001
            status = "EXC"
            nfeat = "-"
            note = str(e)
        print(f"{i:>2}  {str(status):>6}  {str(nfeat):>5}  {label}")
        if note:
            print(f"      -> {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
