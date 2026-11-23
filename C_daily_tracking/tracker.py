"""Local intake records with correction-safe daily aggregation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

from .targets import DailyTargets

CHICAGO = ZoneInfo("America/Chicago")
CORE_NUTRIENTS = ("energy_kcal", "protein_g", "carbohydrate_g", "fat_g")
SUPPORTED_MEALS = {"breakfast", "lunch", "dinner", "snack"}


@dataclass(frozen=True)
class IntakeEntry:
    entry_id: str
    consumed_at: datetime
    meal: str
    nutrients: Mapping[str, float | None]
    consumed_fraction: float = 1.0
    source: str = "manual"
    confirmed: bool = True

    def __post_init__(self) -> None:
        if not self.entry_id.strip():
            raise ValueError("entry_id must not be empty")
        if self.consumed_at.tzinfo is None:
            raise ValueError("consumed_at must be timezone-aware")
        if self.meal not in SUPPORTED_MEALS:
            raise ValueError(f"meal must be one of {sorted(SUPPORTED_MEALS)}")
        if self.consumed_fraction < 0:
            raise ValueError("consumed_fraction must be non-negative")
        unknown = set(self.nutrients) - set(CORE_NUTRIENTS) - {"mass_g"}
        if unknown:
            raise ValueError(f"unsupported nutrient(s): {sorted(unknown)}")
        for name, value in self.nutrients.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number or None")


class DailyTracker:
    """SQLite-backed private log; callers choose a path outside the repository."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS intake_entries (
                    entry_id TEXT PRIMARY KEY,
                    consumed_at TEXT NOT NULL,
                    local_date TEXT NOT NULL,
                    meal TEXT NOT NULL,
                    nutrients_json TEXT NOT NULL,
                    consumed_fraction REAL NOT NULL,
                    source TEXT NOT NULL,
                    confirmed INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_intake_local_date "
                "ON intake_entries(local_date)"
            )

    def upsert(self, entry: IntakeEntry) -> None:
        """Insert or replace one logical record, preventing correction double-counting."""

        consumed_chicago = entry.consumed_at.astimezone(CHICAGO)
        now = datetime.now(CHICAGO).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO intake_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    consumed_at=excluded.consumed_at,
                    local_date=excluded.local_date,
                    meal=excluded.meal,
                    nutrients_json=excluded.nutrients_json,
                    consumed_fraction=excluded.consumed_fraction,
                    source=excluded.source,
                    confirmed=excluded.confirmed,
                    updated_at=excluded.updated_at
                """,
                (
                    entry.entry_id,
                    entry.consumed_at.isoformat(),
                    consumed_chicago.date().isoformat(),
                    entry.meal,
                    json.dumps(dict(entry.nutrients), sort_keys=True),
                    entry.consumed_fraction,
                    entry.source,
                    int(entry.confirmed),
                    now,
                ),
            )

    def delete(self, entry_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM intake_entries WHERE entry_id = ?", (entry_id,)
            )
            return cursor.rowcount > 0

    def entries_for_date(self, local_date: date) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM intake_entries WHERE local_date = ? ORDER BY consumed_at",
                (local_date.isoformat(),),
            ).fetchall()
        entries: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["nutrients"] = json.loads(item.pop("nutrients_json"))
            item["confirmed"] = bool(item["confirmed"])
            entries.append(item)
        return entries

    def daily_summary(self, local_date: date, targets: DailyTargets) -> dict[str, object]:
        """Return D-compatible totals plus explicit completeness information."""

        entries = [entry for entry in self.entries_for_date(local_date) if entry["confirmed"]]
        known_totals = {name: 0.0 for name in CORE_NUTRIENTS}
        unknown_counts = {name: 0 for name in CORE_NUTRIENTS}
        for entry in entries:
            nutrients = entry["nutrients"]
            fraction = float(entry["consumed_fraction"])
            for name in CORE_NUTRIENTS:
                value = nutrients.get(name)
                if value is None:
                    unknown_counts[name] += 1
                else:
                    known_totals[name] += float(value) * fraction

        totals = {
            name: (None if unknown_counts[name] else round(value, 2))
            for name, value in known_totals.items()
        }
        remaining = {
            "energy_kcal": _remaining(targets.energy_kcal, totals["energy_kcal"]),
            "protein_g": _remaining(targets.protein_g, totals["protein_g"]),
            "carbohydrate_range_g": _remaining_range(
                targets.carbohydrate_range_g, totals["carbohydrate_g"]
            ),
            "fat_range_g": _remaining_range(targets.fat_range_g, totals["fat_g"]),
        }
        return {
            "date": local_date.isoformat(),
            "timezone": str(CHICAGO),
            "daily_totals": totals,
            "known_totals": {key: round(value, 2) for key, value in known_totals.items()},
            "remaining": remaining,
            "targets": targets.as_dict(),
            "unknown_nutrients": [key for key, count in unknown_counts.items() if count],
            "unknown_entry_counts": unknown_counts,
            "incomplete_log": any(unknown_counts.values()),
            "confirmed_entry_count": len(entries),
        }


def _remaining(target: float, total: object) -> float | None:
    return None if total is None else round(target - float(total), 2)


def _remaining_range(
    bounds: tuple[float, float], total: object
) -> list[float] | None:
    if total is None:
        return None
    return [round(bounds[0] - float(total), 2), round(bounds[1] - float(total), 2)]
