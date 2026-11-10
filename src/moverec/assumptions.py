"""Load and validate config/assumptions.yaml.

Every assumption is a block with a value and a documented source. Validation
fails loudly if a block is missing its source, so undocumented numbers can't
sneak into the generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "assumptions.yaml"

SOURCE_TYPES = {"public", "generic_reasoning", "user_spec", "illustrative"}
REQUIRED_KEYS = {"value", "source_type", "source"}


class AssumptionError(ValueError):
    pass


@dataclass(frozen=True)
class Assumptions:
    meta: dict
    blocks: dict

    def __getitem__(self, name: str):
        """Return the `value` of an assumption block."""
        return self.blocks[name]["value"]

    def needs_attention(self) -> list[tuple[str, str]]:
        """Blocks whose source still has to be checked, as (name, reason)."""
        out = []
        for name, block in self.blocks.items():
            if block["source_type"] == "public" and not block["verified"]:
                out.append((name, "public source not yet verified"))
            if block.get("needs_public_source"):
                out.append((name, "from the project spec; needs a public source or removal before publishing"))
        return out


def validate(raw: dict) -> None:
    if "assumptions" not in raw or not isinstance(raw["assumptions"], dict):
        raise AssumptionError("file must have a top-level 'assumptions' mapping")
    for name, block in raw["assumptions"].items():
        if not isinstance(block, dict):
            raise AssumptionError(f"{name}: must be a mapping")
        missing = REQUIRED_KEYS - block.keys()
        if missing:
            raise AssumptionError(f"{name}: missing {sorted(missing)}")
        if block["source_type"] not in SOURCE_TYPES:
            raise AssumptionError(f"{name}: unknown source_type {block['source_type']!r}")
        if not str(block["source"]).strip():
            raise AssumptionError(f"{name}: empty source")
        if block["source_type"] == "public" and not isinstance(block.get("verified"), bool):
            raise AssumptionError(f"{name}: public sources need verified: true/false")


def load_assumptions(path: str | Path = DEFAULT_PATH) -> Assumptions:
    with open(path) as f:
        raw = yaml.safe_load(f)
    validate(raw)
    return Assumptions(meta=raw.get("meta", {}), blocks=raw["assumptions"])
