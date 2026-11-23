from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from C_daily_tracking import DailyTracker, IntakeEntry, Profile, calculate_reference_targets
from D_meal_planning import plan_from_component_c


class ComponentCDIntegrationTests(unittest.TestCase):
    def test_c_summary_and_menu_feed_d_without_field_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tracker = DailyTracker(Path(temporary) / "intake.sqlite3")
            targets = calculate_reference_targets(Profile(30, 180, 80, "male", "active"))
            tracker.upsert(
                IntakeEntry(
                    "breakfast-1",
                    datetime(2026, 11, 22, 14, tzinfo=timezone.utc),
                    "breakfast",
                    {
                        "energy_kcal": 500,
                        "protein_g": 25,
                        "carbohydrate_g": 60,
                        "fat_g": 15,
                    },
                )
            )
            summary = tracker.daily_summary(date(2026, 11, 22), targets)

        menu = {
            "date": "2026-11-22",
            "meal": "dinner",
            "items": [
                {
                    "id": "gordon-1",
                    "name": "Rice Bowl",
                    "station": "Global Kitchen",
                    "serving_size": {"amount": "1", "unit": "bowl"},
                    "nutrients": {
                        "energy_kcal": 600,
                        "protein_g": 30,
                        "carbohydrate_g": 80,
                        "fat_g": 20,
                    },
                    "complete_for_planning": True,
                },
                {
                    "id": "gordon-incomplete",
                    "name": "Unknown Side",
                    "station": "Sides",
                    "serving_size": {"amount": "1", "unit": "serving"},
                    "nutrients": {
                        "energy_kcal": 100,
                        "protein_g": None,
                        "carbohydrate_g": 20,
                        "fat_g": 1,
                    },
                    "complete_for_planning": False,
                },
            ],
        }

        result = plan_from_component_c(menu, summary, top_k=3)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["counts"]["eligible_foods"], 1)
        self.assertEqual(result["counts"]["incomplete_nutrition"], 1)
        self.assertEqual(result["rankings"]["joint"][0]["items"][0]["item_id"], "gordon-1")
        self.assertEqual(
            result["rankings"]["joint"][0]["projected_daily_totals"]["energy_kcal"],
            500 + 600 * result["rankings"]["joint"][0]["items"][0]["servings"],
        )

    def test_incomplete_c_log_warning_reaches_d(self) -> None:
        summary = {
            "daily_totals": {
                "energy_kcal": 400,
                "protein_g": None,
                "carbohydrate_g": 50,
                "fat_g": 10,
            },
            "targets": {
                "energy_kcal": 2000,
                "protein_g": 80,
                "carbohydrate_range_g": [225, 325],
                "fat_range_g": [44, 78],
            },
            "incomplete_log": True,
        }
        menu = {
            "items": [
                {
                    "id": "complete-1",
                    "name": "Complete Item",
                    "nutrients": {
                        "energy_kcal": 500,
                        "protein_g": 30,
                        "carbohydrate_g": 60,
                        "fat_g": 20,
                    },
                }
            ]
        }
        result = plan_from_component_c(menu, summary, top_k=1)
        self.assertTrue(any("incomplete" in warning.lower() for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
