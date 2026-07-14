"""ArcGIS Feature/Map service client.

Everything here is defensive: services rename, move, and change field names
constantly across Florida county GIS shops. The client:

- probes each candidate layer URL with `?f=json` until one answers
- caches metadata and results on disk (JSON + GeoJSON)
- resolves canonical field names via `fields.resolve_from_metadata`
- pages through results using resultOffset/resultRecordCount
- retries transient failures with exponential backoff

Geometry is always requested in EPSG:4326 (lon/lat) so downstream code
can pick its own projected CRS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from . import fields as F

log = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    pass


@dataclass
class Layer:
    """A resolved feature/map service layer."""
    url: str
    metadata: dict
    field_map: F.FieldMap
    supports_pagination: bool
    max_record_count: int

    def field_type(self, canonical: str) -> str | None:
        """Esri field type (e.g. esriFieldTypeString) for a canonical name.

        Returns None if the canonical name isn't in the field_map or the
        actual column can't be found in the layer metadata."""
        actual = self.field_map.get(canonical)
        if actual is None:
            return None
        if actual.startswith("__COMPOSITE__:"):
            return None
        for f in self.metadata.get("fields") or []:
            if f.get("name") == actual:
                return f.get("type")
        return None

    def is_string_field(self, canonical: str) -> bool:
        """True if the resolved field is Esri string-typed and needs quotes."""
        t = self.field_type(canonical)
        return t == "esriFieldTypeString"


class ArcGISClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_s: float = 60,
        max_retries: int = 4,
        page_size: int = 2000,
        session: requests.Session | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.page_size = page_size
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "church-scanner/1.0 (Florida DOR parcel research)"
        })

    # ---- HTTP core -----------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_s)
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError as e:
                        raise ServiceError(
                            f"non-JSON response from {url}: {e}"
                        ) from e
                    if isinstance(data, dict) and "error" in data:
                        # ArcGIS often returns 200 with an error body — surface
                        # the details we usually need (invalid where, bad
                        # field name, etc.) plus the request that caused it.
                        self._log_failed_request(url, params, data.get("error"))
                        raise ServiceError(
                            f"service error at {url}: {data['error']}"
                        )
                    return data
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.RequestException(
                        f"transient {r.status_code} from {url}"
                    )
                # Terminal 4xx: dump the request so the user can see what
                # was sent and diagnose without adding print statements.
                self._log_failed_request(url, params, r.text[:500])
                raise ServiceError(f"HTTP {r.status_code} from {url}")
            except (requests.RequestException, ServiceError) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                log.warning(
                    "retry %d/%d for %s: %s",
                    attempt + 1, self.max_retries, url, e,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise ServiceError(f"giving up on {url}: {last_exc}")

    @staticmethod
    def _log_failed_request(url: str, params: dict | None, err) -> None:
        p = dict(params or {})
        log.error("REQUEST FAILED: %s", url)
        for k in ("f", "where", "outFields", "outSR", "returnGeometry",
                  "geometryType", "spatialRel", "resultOffset",
                  "resultRecordCount"):
            if k in p:
                log.error("  %s = %s", k, p[k])
        if "geometry" in p:
            g = p["geometry"]
            log.error("  geometry = %s...", str(g)[:120])
        log.error("  response error = %s", err)

    # ---- Layer resolution ---------------------------------------------

    def resolve_layer(self, candidate_urls: Iterable[str]) -> Layer:
        """Try each candidate URL in order; return the first that responds."""
        errors: list[str] = []
        for url in candidate_urls:
            try:
                meta = self._get_layer_metadata(url)
                fm = F.resolve_from_metadata(url, meta)
                supports_pag = bool(meta.get("advancedQueryCapabilities", {}).get(
                    "supportsPagination", meta.get("supportsPagination", True)
                ))
                max_rc = int(meta.get("maxRecordCount", self.page_size))
                log.info("layer resolved: %s (fields mapped: %d, unresolved: %s)",
                         url, len(fm.mapping), fm.unresolved)
                return Layer(
                    url=url, metadata=meta, field_map=fm,
                    supports_pagination=supports_pag,
                    max_record_count=max_rc,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("candidate layer failed (%s): %s", url, e)
                errors.append(f"{url}: {e}")
        raise ServiceError("no candidate layer responded. attempts:\n  " +
                           "\n  ".join(errors))

    def _get_layer_metadata(self, layer_url: str) -> dict:
        cache_key = self._cache_key(layer_url, ".meta.json")
        if cache_key.exists():
            try:
                return json.loads(cache_key.read_text())
            except Exception:  # noqa: BLE001
                cache_key.unlink(missing_ok=True)
        data = self._get(layer_url, params={"f": "json"})
        if "fields" not in data:
            raise ServiceError(
                f"layer metadata at {layer_url} has no `fields` array"
            )
        cache_key.write_text(json.dumps(data))
        return data

    # ---- Query --------------------------------------------------------

    def query_all(
        self,
        layer: Layer,
        where: str = "1=1",
        out_fields: list[str] | None = None,
        geometry: dict | None = None,
        geometry_type: str = "esriGeometryEnvelope",
        out_sr: int = 4326,
        return_geometry: bool = True,
    ) -> list[dict]:
        """Page through all features matching `where`."""
        page_size = min(self.page_size, layer.max_record_count or self.page_size)
        of_str = "*" if not out_fields else ",".join(sorted(set(out_fields)))
        base_params: dict[str, Any] = {
            "f": "geojson" if return_geometry else "json",
            "where": where,
            "outFields": of_str,
            "outSR": out_sr,
            "returnGeometry": "true" if return_geometry else "false",
        }
        if geometry is not None:
            base_params["geometry"] = json.dumps(geometry)
            base_params["geometryType"] = geometry_type
            base_params["spatialRel"] = "esriSpatialRelIntersects"
            base_params["inSR"] = 4326

        features: list[dict] = []
        offset = 0
        # Log the outgoing query once (subsequent pages just increment offset).
        log.info(
            "QUERY %s/query where=%r outFields=%s returnGeometry=%s "
            "geometryType=%s pageSize=%d",
            layer.url, where, of_str, return_geometry,
            base_params.get("geometryType", "n/a"), page_size,
        )
        while True:
            params = dict(base_params)
            params["resultOffset"] = offset
            params["resultRecordCount"] = page_size
            data = self._get(f"{layer.url}/query", params=params)
            batch = data.get("features", []) or []
            features.extend(batch)
            log.info(
                "  page: %s+%d (offset %d, total %d)",
                layer.url.split("/services/")[-1], len(batch), offset,
                len(features),
            )
            if len(batch) < page_size:
                break
            if not layer.supports_pagination:
                log.warning("layer lacks pagination — stopping after first page")
                break
            offset += page_size
            if offset > 500_000:
                raise ServiceError(
                    "aborting: exceeded 500k features — geometry filter is "
                    "probably too broad"
                )
        return features

    # ---- Cache helpers ------------------------------------------------

    def cache_get_geojson(self, key: str) -> dict | None:
        p = self.cache_dir / f"{key}.geojson"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                p.unlink(missing_ok=True)
        return None

    def cache_put_geojson(self, key: str, features: list[dict]) -> Path:
        p = self.cache_dir / f"{key}.geojson"
        fc = {"type": "FeatureCollection", "features": features}
        p.write_text(json.dumps(fc))
        return p

    def _cache_key(self, url: str, suffix: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}{suffix}"
