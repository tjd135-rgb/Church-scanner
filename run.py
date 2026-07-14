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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("run")

    cfg = cfg_mod.load(args.config)
    log.info("loaded config from %s", cfg.path)

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


if __name__ == "__main__":
    sys.exit(main())
