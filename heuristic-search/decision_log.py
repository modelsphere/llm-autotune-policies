"""Append-only local decision records; no platform traffic."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class DecisionLog:
    """Write and flush each event immediately, sharing a unique ID per run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.run_id = uuid4().hex

    def __call__(self, event: dict) -> None:
        record = {**event, "schema_version": 1, "run_id": self.run_id,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_json_safe(record), ensure_ascii=False,
                                        allow_nan=False) + "\n")
        except OSError:
            logging.getLogger(__name__).exception("Cannot write decision log %s", self.path)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
