"""Logging helpers scoped to the orbit simulator package."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


class RunLogger:
    """Buffered logger that stores simulation data to CSV files."""

    TIMESERIES_HEADER = [
        "t",
        "x",
        "y",
        "vx",
        "vy",
        "r",
        "v",
        "energy",
        "h",
        "e",
        "dt_eff",
    ]
    EVENTS_HEADER = ["t", "type", "r", "v", "details"]

    def __init__(
        self,
        root_dir: str | Path = "data/runs",
        run_id: Optional[str] = None,
        *,
        timeseries_flush_threshold: int = 200,
        events_flush_threshold: int = 50,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def make_candidate(suffix: Optional[int] = None) -> str:
            base = run_id or f"{timestamp}_run"
            if suffix is None:
                return base
            if run_id:
                return f"{run_id}_{suffix}"
            return f"{base}_{suffix:02d}"

        candidate_id = make_candidate()
        suffix = 1
        while (self.root_dir / candidate_id).exists():
            candidate_id = make_candidate(suffix)
            suffix += 1

        self.run_id = candidate_id
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.timeseries_path = self.run_dir / "timeseries.csv"
        self.events_path = self.run_dir / "events.csv"
        self.meta_path = self.run_dir / "meta.json"

        self._ts_file = self.timeseries_path.open("w", newline="")
        self._ts_file.write(",".join(self.TIMESERIES_HEADER) + "\n")
        self._ev_file = self.events_path.open("w", newline="")
        self._ev_file.write(",".join(self.EVENTS_HEADER) + "\n")

        self._ts_buffer: list[str] = []
        self._ev_buffer: list[str] = []
        self._ts_threshold = max(1, timeseries_flush_threshold)
        self._ev_threshold = max(1, events_flush_threshold)

        last_run_marker = self.root_dir / "last_run.txt"
        last_run_marker.write_text(self.run_id, encoding="utf-8")

    def write_meta(self, meta: dict) -> None:
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    def log_ts(self, values: Sequence[float]) -> None:
        self._ts_buffer.append(",".join(self._format_value(v) for v in values))
        if len(self._ts_buffer) >= self._ts_threshold:
            self._flush_timeseries()

    def log_event(self, values: Sequence[object]) -> None:
        self._ev_buffer.append(",".join(self._format_event_value(v) for v in values))
        if len(self._ev_buffer) >= self._ev_threshold:
            self._flush_events()

    def close(self) -> None:
        self._flush_timeseries()
        self._flush_events()
        self._ts_file.close()
        self._ev_file.close()

    def _flush_timeseries(self) -> None:
        if self._ts_buffer:
            self._ts_file.write("\n".join(self._ts_buffer) + "\n")
            self._ts_file.flush()
            self._ts_buffer.clear()

    def _flush_events(self) -> None:
        if self._ev_buffer:
            self._ev_file.write("\n".join(self._ev_buffer) + "\n")
            self._ev_file.flush()
            self._ev_buffer.clear()

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:.10g}"

    @staticmethod
    def _format_event_value(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.10g}"
        return str(value)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None


__all__ = ["RunLogger"]
