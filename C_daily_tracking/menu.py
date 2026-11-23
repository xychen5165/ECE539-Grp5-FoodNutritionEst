"""Dated Gordon Nutrislice menu retrieval and normalization."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

CHICAGO = ZoneInfo("America/Chicago")
MEAL_TYPES = {"breakfast", "lunch", "dinner"}
NUTRIENT_FIELDS = {
    "energy_kcal": "calories",
    "protein_g": "g_protein",
    "carbohydrate_g": "g_carbs",
    "fat_g": "g_fat",
}


class MenuDataError(ValueError):
    """Raised when a remote menu violates the Component C contract."""


class GordonMenuClient:
    BASE_URL = (
        "https://wisc-housingdining.api.nutrislice.com/menu/api/weeks/"
        "school/gordon-avenue-market/menu-type"
    )

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_cache_age: timedelta = timedelta(hours=6),
        fetch_json: Callable[[str], object] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age = max_cache_age
        self.fetch_json = fetch_json or _fetch_json

    def get_menu(
        self, menu_date: date, meal: str, *, force_refresh: bool = False
    ) -> dict[str, object]:
        if meal not in MEAL_TYPES:
            raise ValueError(f"meal must be one of {sorted(MEAL_TYPES)}")
        cache_path = self.cache_dir / f"{menu_date.isoformat()}-{meal}.json"
        now = datetime.now(CHICAGO)
        if not force_refresh and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if now - fetched_at <= self.max_cache_age:
                result = cached["menu"]
                result["cache_status"] = "hit"
                return result

        url = self._url(menu_date, meal)
        payload = self.fetch_json(url)
        result = self._normalize(payload, menu_date, meal, url, now)
        envelope = {"fetched_at": now.isoformat(), "menu": result}
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)
        return result

    def _url(self, menu_date: date, meal: str) -> str:
        return (
            f"{self.BASE_URL}/{meal}/{menu_date.year}/"
            f"{menu_date.month:02d}/{menu_date.day:02d}/"
        )

    @staticmethod
    def _normalize(
        payload: object,
        menu_date: date,
        meal: str,
        source_url: str,
        fetched_at: datetime,
    ) -> dict[str, object]:
        if not isinstance(payload, dict) or not isinstance(payload.get("days"), list):
            raise MenuDataError("Nutrislice response must contain a days list")
        day = next(
            (item for item in payload["days"] if item.get("date") == menu_date.isoformat()),
            None,
        )
        if day is None:
            raise MenuDataError(f"response does not contain requested date {menu_date}")

        items: list[dict[str, object]] = []
        station: str | None = None
        for raw in day.get("menu_items", []):
            if raw.get("is_station_header") or raw.get("is_section_title"):
                station = raw.get("text") or station
                continue
            food = raw.get("food")
            if not isinstance(food, dict):
                continue
            nutrition = food.get("rounded_nutrition_info") or {}
            nutrients = {
                output_name: _validated_number(nutrition.get(remote_name), output_name)
                for output_name, remote_name in NUTRIENT_FIELDS.items()
            }
            serving = food.get("serving_size_info") or {}
            amount = serving.get("serving_size_amount", raw.get("serving_size_amount"))
            unit = serving.get("serving_size_unit", raw.get("serving_size_unit"))
            items.append(
                {
                    "id": str(food.get("id", raw.get("id"))),
                    "name": str(food.get("name", "")).strip(),
                    "station": station or raw.get("category") or food.get("food_category"),
                    "serving_size": {"amount": amount, "unit": unit},
                    "nutrients": nutrients,
                    "complete_for_planning": all(value is not None for value in nutrients.values()),
                }
            )
        return {
            "date": menu_date.isoformat(),
            "meal": meal,
            "timezone": str(CHICAGO),
            "fetched_at": fetched_at.isoformat(),
            "source_url": source_url,
            "cache_status": "miss",
            "items": items,
        }


def _validated_number(value: object, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MenuDataError(f"{name} must be numeric and use the expected kcal/g unit")
    if value < 0:
        raise MenuDataError(f"{name} must be non-negative")
    return float(value)


def _fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "ECE539-FoodNutritionEst/1.0"})
    with urlopen(request, timeout=20) as response:  # nosec B310: fixed HTTPS base URL
        return json.load(response)
