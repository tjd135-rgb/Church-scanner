"""Configuration loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def buffer_ft(self) -> float:
        return float(self.raw.get("buffer_ft", 750))

    @property
    def cities(self) -> list[dict[str, str]]:
        return list(self.raw.get("cities", []))

    @property
    def roads(self) -> list[dict[str, Any]]:
        return list(self.raw.get("roads", []))

    @property
    def address_patterns(self) -> list[str]:
        return [p.upper() for p in self.raw.get("address_patterns", [])]

    @property
    def dor_use_codes(self) -> list[int]:
        return [int(c) for c in self.raw.get("dor_use_codes", [71, 72])]

    @property
    def score_weights(self) -> dict[str, float]:
        return dict(self.raw.get("score_weights", {}))

    @property
    def score_caps(self) -> dict[str, float]:
        return dict(self.raw.get("score_caps", {}))

    @property
    def validation_parcels(self) -> list[dict[str, Any]]:
        return list(self.raw.get("validation_parcels", []))

    @property
    def sources(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw.get("sources", {}))

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.raw.get("runtime", {}))

    @property
    def cache_dir(self) -> Path:
        return Path(self.runtime.get("cache_dir", "./data/cache")).resolve()


def load(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, path=p)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    w = cfg.score_weights
    total = sum(w.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"score_weights must sum to 1.0 (got {total:.4f}) in {cfg.path}"
        )
    for k in ("acreage", "lur_inverse", "land_value_share", "hold_period"):
        if k not in w:
            raise ValueError(f"score_weights missing key: {k}")
