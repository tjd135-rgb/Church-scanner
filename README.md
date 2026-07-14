# Church Scanner

Identifies church (Florida DOR use code 71) and private-school
(DOR use code 72) parcels along the **US-1 (Federal Highway)** and
**Dixie Highway / SR 811** corridor from **Boca Raton** through
**Deerfield Beach** to **Pompano Beach**, and scores them by
redevelopment attractiveness (land underutilization, dirt-heavy
valuation, long ownership).

Produces a sortable CSV, a formatted Excel workbook (Summary +
Parcels + Top 10 tabs), and a GeoJSON for GIS re-use — all from a
single command.

## Install

```
pip install -r requirements.txt
```

## Run

```
python run.py
```

Defaults:

* Reads `./config.yaml`.
* Writes `./output/church_school_parcels.{csv,xlsx,geojson}`.
* Caches network responses in `./data/cache/`.
* Buffer around road centerlines: **750 ft**. Override with
  `--buffer-ft 1000`.
* Fails with exit code 2 if any of the four known validation parcels
  (St. Ambrose, Deerfield Christian Academy, Cathedral Church of God,
  Florida Coast Church) is missing. Pass `--skip-validation` to
  export anyway.

## What it does

1. **Resolves a Florida parcel Feature Service.** Tries the FGIO
   statewide layer first, then falls back to Broward BCPA and Palm
   Beach PAPA. Every layer's `?f=json` metadata is inspected so field
   names are mapped at runtime — you don't have to edit code when a
   county renames a column.
2. **Loads city polygons** for Boca Raton, Deerfield Beach, and
   Pompano Beach. Falls back to a corridor bounding box if the
   municipal boundary layer is unreachable.
3. **Pulls road centerlines** for US-1 and Dixie Highway from
   OpenStreetMap via `osmnx`, and buffers them by 750 ft to define
   the corridor polygon.
4. **Runs three extraction passes** and unions the results:
   * **A — Land-use filter.** Parcels with DOR_UC in {71, 72}, clipped
     to the city polygons.
   * **B — Spatial buffer.** Parcels intersecting the buffered
     corridor. DOR_UC filter re-applied server-side to keep the query
     cheap.
   * **C — Situs-address regex.** Catches parcels whose address
     matches "Federal Hwy", "US-1", "Dixie Hwy", "SR 811", etc. but
     that the buffer missed because of imprecise centerlines.
5. **Fills / normalizes fields**: acres from square feet, square feet
   from acres, and both from geometry area as a last resort. Numeric
   coercion on valuations and living-area columns.
6. **Computes metrics**:
   * `land_utilization_ratio = building_sf / lot_sf`
   * `land_value_share = land_val / total_val`
   * `hold_period_years = years since last sale`
7. **Composite redevelopment score (0–100)** — weighted sum of four
   0–100 sub-scores. Weights are declared in `config.yaml` and echoed
   on the Summary tab of the workbook so nothing is a black box.
8. **Validates against four known parcels** and reports which passes
   caught them.
9. **Exports** CSV + XLSX (Summary / Parcels / Top 10 tabs) +
   GeoJSON.

## Configuration

Every knob lives in `config.yaml`:

* `buffer_ft` — corridor buffer width.
* `cities` — the three target cities.
* `roads` — OSM `ref` / `name` patterns matched to build the
  corridor centerlines.
* `address_patterns` — situs address regex list for pass C.
* `dor_use_codes` — defaults to `[71, 72]`.
* `score_weights` — must sum to 1.0.
* `score_caps` — where each sub-score saturates.
* `sources.*.layers` — ordered ArcGIS REST URLs to try. Add or
  reorder freely; the client picks the first one that returns valid
  metadata.
* `runtime.cache_dir` — persistent cache of service metadata + OSM
  responses so repeated runs are near-free.

## Scoring model

```
score_acreage           = clip(lot_ac / acreage_ac_max, 0, 1) * 100
score_lur_inverse       = clip((1 - (LUR - lur_zero_target) / (1 - lur_zero_target)) * 100, 0, 100)
score_land_value_share  = clip(LVS / land_value_share_max, 0, 1) * 100
score_hold_period       = clip(years_since_sale / hold_period_years_max, 0, 1) * 100
redevelopment_score     = w_acr*acr + w_lur*lur + w_lvs*lvs + w_hold*hold
```

Missing LUR or LVS contribute a neutral 50 sub-score so data gaps
don't unfairly bury or reward rows. Parcels with no sale on record
are treated as very-long-hold and get the full 100 for hold_period.

Adjust weights in `config.yaml` and re-run — the scoring model is not
compiled into any hard-coded constants.

## Repo layout

```
run.py                     # single-command entry
config.yaml                # every knob
requirements.txt
scanner/
  __init__.py
  config.py                # config loader + weight-sum validation
  fields.py                # canonical -> source-column alias map
  sources.py               # ArcGIS REST client (metadata, paging, retry)
  roads.py                 # OSM road centerlines + city polygons
  extract.py               # 3-pass extraction + merge
  metrics.py               # LUR / LVS / hold + composite score
  validate.py              # 4 known-parcel check
  export.py                # CSV + XLSX (Summary / Parcels / Top 10)
  dor.py                   # DOR use-code lookup
  pipeline.py              # end-to-end orchestration
```

## Notes / caveats

* This tool is a screen, not an underwriting model. The composite
  score prioritizes parcels worth a conversation, not a bid.
* Building SF from FGDL is the "adjusted gross" or "total living
  area" — not always identical to what a county reports as gross
  building SF. LUR is a rough proxy, not a coverage ratio.
* Some parcels legitimately have `total_val = 0` (public exemptions,
  data lag). LUR / LVS handle these gracefully by treating them as
  missing.
* Cached metadata lives under `data/cache/`. Delete that directory to
  force a fresh pull if a service was updated.
