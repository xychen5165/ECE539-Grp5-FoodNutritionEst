from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.daily_tracking import (
    DailyTracker,
    GordonMenuClient,
    IntakeEntry,
    MenuDataError,
    Profile,
    calculate_reference_targets,
)


class TargetTests(unittest.TestCase):
    def test_male_active_2023_eer_and_ranges(self) -> None:
        profile = Profile(30, 180, 80, "male", "active")
        targets = calculate_reference_targets(profile)
        expected = 1004.82 - 10.83 * 30 + 6.52 * 180 + 15.91 * 80
        self.assertAlmostEqual(targets.energy_kcal, expected, places=2)
        self.assertEqual(targets.protein_g, 64.0)
        self.assertEqual(targets.carbohydrate_range_g[0], round(expected * 0.45 / 4, 2))
        self.assertEqual(targets.fat_range_g[1], round(expected * 0.35 / 9, 2))

    def test_explicit_adjustments_and_overrides(self) -> None:
        profile = Profile(25, 165, 60, "female", "low_active")
        base = calculate_reference_targets(profile)
        edited = calculate_reference_targets(
            profile,
            energy_adjustment_kcal=-200,
            protein_g_per_kg=1.2,
            overrides={"protein_g": 75.0},
        )
        self.assertAlmostEqual(edited.energy_kcal, base.energy_kcal - 200)
        self.assertEqual(edited.protein_g, 75.0)
        self.assertIn("user-confirmed", edited.source)


class TrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.tracker = DailyTracker(Path(self.temp.name) / "intake.sqlite3")
        self.targets = calculate_reference_targets(Profile(30, 180, 80, "male", "active"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fraction_and_correction_replace_instead_of_double_count(self) -> None:
        when = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
        nutrients = {
            "energy_kcal": 600,
            "protein_g": 30,
            "carbohydrate_g": 80,
            "fat_g": 20,
        }
        self.tracker.upsert(IntakeEntry("meal-1", when, "lunch", nutrients, 0.5))
        first = self.tracker.daily_summary(date(2026, 11, 22), self.targets)
        self.assertEqual(first["daily_totals"]["energy_kcal"], 300)

        self.tracker.upsert(IntakeEntry("meal-1", when, "lunch", nutrients, 0.75))
        corrected = self.tracker.daily_summary(date(2026, 11, 22), self.targets)
        self.assertEqual(corrected["confirmed_entry_count"], 1)
        self.assertEqual(corrected["daily_totals"]["energy_kcal"], 450)

    def test_unknown_nutrient_marks_log_incomplete_without_assuming_zero(self) -> None:
        self.tracker.upsert(
            IntakeEntry(
                "manual-1",
                datetime(2026, 11, 22, 9, tzinfo=timezone.utc),
                "breakfast",
                {
                    "energy_kcal": 300,
                    "protein_g": None,
                    "carbohydrate_g": 40,
                    "fat_g": 10,
                },
            )
        )
        summary = self.tracker.daily_summary(date(2026, 11, 22), self.targets)
        self.assertTrue(summary["incomplete_log"])
        self.assertIsNone(summary["daily_totals"]["protein_g"])
        self.assertIsNone(summary["remaining"]["protein_g"])
        self.assertEqual(summary["known_totals"]["protein_g"], 0)

    def test_chicago_date_controls_rollover(self) -> None:
        # 01:00 UTC on Nov 23 is still Nov 22 in Chicago (UTC-6).
        self.tracker.upsert(
            IntakeEntry(
                "late-1",
                datetime(2026, 11, 23, 1, tzinfo=timezone.utc),
                "dinner",
                {name: 1 for name in ("energy_kcal", "protein_g", "carbohydrate_g", "fat_g")},
            )
        )
        self.assertEqual(len(self.tracker.entries_for_date(date(2026, 11, 22))), 1)
        self.assertEqual(len(self.tracker.entries_for_date(date(2026, 11, 23))), 0)

    def test_unconfirmed_entry_is_not_counted(self) -> None:
        self.tracker.upsert(
            IntakeEntry(
                "draft-1",
                datetime(2026, 11, 22, 12, tzinfo=timezone.utc),
                "snack",
                {name: 100 for name in ("energy_kcal", "protein_g", "carbohydrate_g", "fat_g")},
                confirmed=False,
            )
        )
        summary = self.tracker.daily_summary(date(2026, 11, 22), self.targets)
        self.assertEqual(summary["confirmed_entry_count"], 0)


class MenuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.calls: list[str] = []
        self.payload = {
            "days": [
                {
                    "date": "2026-11-22",
                    "menu_items": [
                        {"is_station_header": True, "text": "Global Kitchen", "food": None},
                        {
                            "id": 7,
                            "food": {
                                "id": 42,
                                "name": "Rice Bowl",
                                "rounded_nutrition_info": {
                                    "calories": 500,
                                    "g_protein": 20,
                                    "g_carbs": 70,
                                    "g_fat": 15,
                                },
                                "serving_size_info": {
                                    "serving_size_amount": "1",
                                    "serving_size_unit": "bowl",
                                },
                            },
                        },
                    ],
                }
            ]
        }

        def fetch(url: str) -> object:
            self.calls.append(url)
            return self.payload

        self.client = GordonMenuClient(
            self.temp.name, max_cache_age=timedelta(days=1), fetch_json=fetch
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_normalization_and_cache_are_scoped_by_meal(self) -> None:
        day = date(2026, 11, 22)
        lunch = self.client.get_menu(day, "lunch")
        cached_lunch = self.client.get_menu(day, "lunch")
        dinner = self.client.get_menu(day, "dinner")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(cached_lunch["cache_status"], "hit")
        self.assertEqual(dinner["meal"], "dinner")
        self.assertEqual(lunch["items"][0]["station"], "Global Kitchen")
        self.assertTrue(lunch["items"][0]["complete_for_planning"])

    def test_bad_nutrient_unit_shape_is_rejected(self) -> None:
        self.payload["days"][0]["menu_items"][1]["food"]["rounded_nutrition_info"][
            "g_fat"
        ] = "15 mg"
        with self.assertRaises(MenuDataError):
            self.client.get_menu(date(2026, 11, 22), "breakfast")


if __name__ == "__main__":
    unittest.main()
