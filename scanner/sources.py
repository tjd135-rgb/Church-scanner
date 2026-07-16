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
    """Base for all service-side errors."""
    pass


class TerminalServiceError(ServiceError):
    """A 4xx or explicit ArcGIS error body that will not succeed on retry.

    Distinguishing terminal errors from transient ones (5xx, 429, network
    timeouts) matters because retrying a hard 400 four times just wastes
    the user's minutes and clogs the log. Terminal errors bubble up
    immediately; transient ones use exponential backoff."""
    pass


@dataclass
class Layer:
    """A resolved feature/map service layer."""
    url: str
    metadata: dict
    field_map: F.FieldMap
    supports_pagination: bool
    max_record_count: int

    @property
    def supports_geojson_output(self) -> bool:
        """Does this layer advertise geoJSON in its supportedQueryFormats?

        Large enterprise FeatureServer instances (e.g. FGIO Statewide
        Cadastral, ~10M parcels) commonly restrict output to `JSON,AMF`
        and reject f=geojson with a generic 'Unable to perform query.
        Please check your parameters.' 400 — the same message the user's
        run produced. We look at the metadata rather than trial-and-
        erroring the query."""
        fmts = str(self.metadata.get("supportedQueryFormats") or "")
        return "geojson" in fmts.lower()

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
                        # ArcGIS often returns 200 with an error body. These
                        # are terminal — no amount of retrying will change
                        # 'field not found' or 'invalid URL'.
                        self._log_failed_request(url, params, data.get("error"))
                        raise TerminalServiceError(
                            f"service error at {url}: {data['error']}"
                        )
                    return data
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.RequestException(
                        f"transient {r.status_code} from {url}"
                    )
                # Terminal 4xx: dump the request and stop retrying.
                self._log_failed_request(url, params, r.text[:500])
                raise TerminalServiceError(f"HTTP {r.status_code} from {url}")
            except TerminalServiceError:
                # No retry — the response is what it is.
                raise
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
        candidates = list(candidate_urls or [])
        if not candidates:
            raise ServiceError("no candidate URLs configured for this source")
        errors: list[str] = []
        for url in candidates:
            try:
                meta = self._get_layer_metadata(url)
                fm = F.resolve_from_metadata(url, meta)
                supports_pag = bool(meta.get("advancedQueryCapabilities", {}).get(
                    "supportsPagination", meta.get("supportsPagination", True)
                ))
                max_rc = int(meta.get("maxRecordCount", self.page_size))
                log.info(
                    "layer resolved: %s (fields mapped: %d, unresolved: %s, "
                    "supportedQueryFormats=%r)",
                    url, len(fm.mapping), fm.unresolved,
                    meta.get("supportedQueryFormats"),
                )
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
        """Page through all features matching `where`.

        If a polygon geometry filter triggers a 400, we transparently
        retry with the polygon's bounding envelope — a common workaround
        for services that reject valid GeoJSON polygons under some
        combination of spatial-index state, winding, or SR handling.
        """
        page_size = min(self.page_size, layer.max_record_count or self.page_size)
        of_str = "*" if not out_fields else ",".join(sorted(set(out_fields)))
        # Always use Esri JSON (f=json). GeoJSON output is up to ~5x heavier
        # to serialize per feature; on a wide+tall payload (many rows,
        # many fields, with geometry) FGIO Statewide Cadastral rejects it
        # with 'Unable to perform query' even though its metadata
        # advertises geoJSON in supportedQueryFormats. Esri JSON works
        # everywhere and we convert to GeoJSON-shaped features client-side
        # in _esri_features_to_geojson.
        base_params: dict[str, Any] = {
            "f": "json",
            "where": where,
            "outFields": of_str,
            "outSR": out_sr,
            "returnGeometry": "true" if return_geometry else "false",
        }
        if geometry is not None:
            base_params["geometry"] = json.dumps(geometry)
            base_params["geometryType"] = geometry_type
            base_params["spatialRel"] = "esriSpatialRelIntersects"
            # Read inSR from the geometry object so the caller's choice
            # of SR is honored end-to-end. Falls back to 4326 if absent.
            g_sr = (geometry.get("spatialReference") or {}) if isinstance(
                geometry, dict
            ) else {}
            base_params["inSR"] = int(
                g_sr.get("wkid") or g_sr.get("latestWkid") or 4326
            )

        # Log the outgoing query once (subsequent pages just increment offset).
        log.info(
            "QUERY %s/query where=%r outFields=%s returnGeometry=%s "
            "geometryType=%s pageSize=%d",
            layer.url, where, of_str, return_geometry,
            base_params.get("geometryType", "n/a"), page_size,
        )
        if geometry is not None:
            log.debug("  geometry payload: %s", json.dumps(geometry)[:2000])

        try:
            return self._paged(layer, base_params, page_size)
        except ServiceError as e:
            if (
                geometry is not None
                and geometry_type == "esriGeometryPolygon"
                and _is_geometry_error(str(e))
            ):
                env_geom = _polygon_to_envelope(geometry)
                log.warning(
                    "polygon geometry filter 400'd; retrying with bounding "
                    "envelope %s", env_geom,
                )
                retry_params = dict(base_params)
                retry_params["geometry"] = json.dumps(env_geom)
                retry_params["geometryType"] = "esriGeometryEnvelope"
                return self._paged(layer, retry_params, page_size)
            raise

    def _paged(
        self, layer: Layer, base_params: dict, page_size: int,
    ) -> list[dict]:
        features: list[dict] = []
        offset = 0
        is_esri_json = base_params.get("f") == "json"
        while True:
            params = dict(base_params)
            params["resultOffset"] = offset
            params["resultRecordCount"] = page_size
            data = self._get(f"{layer.url}/query", params=params)
            if is_esri_json:
                batch = _esri_features_to_geojson(data)
            else:
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


# --------------------------------------------------------------------
# Module-level helpers for the polygon->envelope retry fallback.
# --------------------------------------------------------------------

_GEOMETRY_ERROR_MARKERS = (
    "unable to perform query",
    "invalid geometry",
    "invalid spatial reference",
    "geometry is malformed",
    "check your parameters",
    "http 400",
)


def _is_geometry_error(err_str: str) -> bool:
    """Best-effort match for ArcGIS errors that suggest a geometry problem
    rather than a WHERE-clause or field-name problem."""
    s = err_str.lower()
    return any(m in s for m in _GEOMETRY_ERROR_MARKERS)


def _polygon_to_envelope(esri_geom: dict) -> dict:
    """Reduce an Esri polygon (rings) to its bounding envelope."""
    rings = esri_geom.get("rings") or []
    if not rings:
        raise ValueError("polygon geometry has no rings")
    xs = [pt[0] for ring in rings for pt in ring]
    ys = [pt[1] for ring in rings for pt in ring]
    return {
        "xmin": min(xs), "ymin": min(ys),
        "xmax": max(xs), "ymax": max(ys),
        "spatialReference": esri_geom.get(
            "spatialReference", {"wkid": 4326}
        ),
    }


def _esri_features_to_geojson(response: dict) -> list[dict]:
    """Convert an Esri JSON query response (features[].attributes +
    features[].geometry.rings/paths/points) into GeoJSON-style Feature
    dicts (properties + geometry) so the extractor's _canonicalize can
    read either format without branching."""
    out: list[dict] = []
    for feat in response.get("features") or []:
        props = feat.get("attributes") or {}
        egeom = feat.get("geometry")
        gjson_geom = _esri_geom_to_geojson(egeom) if egeom else None
        out.append({"type": "Feature", "properties": props,
                    "geometry": gjson_geom})
    return out


def _esri_geom_to_geojson(egeom: dict) -> dict | None:
    if egeom is None:
        return None
    if "x" in egeom and "y" in egeom:
        return {"type": "Point", "coordinates": [egeom["x"], egeom["y"]]}
    if "rings" in egeom:
        rings = egeom["rings"]
        if not rings:
            return None
        # ArcGIS returns exterior rings CW; GeoJSON wants CCW exterior.
        # shapely will accept either, so we don't bother rewinding here —
        # the extractor pushes everything through shapely anyway. If a
        # downstream consumer is strict, run shapely.geometry.polygon.orient.
        if len(rings) == 1:
            return {"type": "Polygon", "coordinates": [rings[0]]}
        return {"type": "Polygon", "coordinates": rings}
    if "paths" in egeom:
        paths = egeom["paths"]
        if len(paths) == 1:
            return {"type": "LineString", "coordinates": paths[0]}
        return {"type": "MultiLineString", "coordinates": paths}
    return None
