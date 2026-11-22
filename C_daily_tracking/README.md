# C - Daily nutrition tracking

Component workspace for reference targets, confirmed intake records and menu access. The reusable implementation is in [`src/daily_tracking`](../src/daily_tracking/).

## Deliverables
- Implement sourced, editable reference targets from profile and goal inputs.
- Apply user-confirmed consumption fractions, record meals/snacks and calculate daily totals and remaining goals.
- Support corrections without double-counting; identify incomplete logs and unknown nutrients.
- Fetch Gordon menus by America/Chicago date and selected breakfast/lunch/dinner; cache dated responses and validate nutrient units.
- Test calculation rules, date rollover, meal switching and record updates.

## Interface
Input: component B nutrient estimates, confirmed consumption, manual records and user profile/goals.
Output: daily totals, remaining targets/bounds and validated Gordon menu data for component D.

Generic meal-photo estimates are independent of Gordon menu recipes. Keep component notes here and reusable code in ../src/. Profiles, food logs and private identity mappings must stay outside this public repository.

## Implemented behavior

- `calculate_reference_targets` supports generally healthy adults age 19+ using the 2023 DRI EER equations, adult carbohydrate/fat AMDRs, and a 0.8 g/kg protein starting value. Goal-specific energy adjustments, exercise protein factors, and final target overrides are explicit user-confirmed inputs.
- `DailyTracker` stores private records in a caller-selected SQLite file. Confirmed fractions are applied during aggregation; reusing an entry ID replaces the earlier version, so corrections do not double-count.
- Daily boundaries use `America/Chicago`. Missing core nutrients produce `None`, an `incomplete_log` flag, and a separate known subtotal instead of silently treating unknown values as zero.
- `GordonMenuClient` requests a selected breakfast/lunch/dinner and date from Nutrislice, keeps meal/date-specific timestamped caches, normalizes kcal/g fields, and marks incomplete foods as unavailable for complete planning.

These values are planning references, not medical advice. Pregnancy, breastfeeding, people under 19, and clinical nutrition needs require a different workflow.

## Minimal example

```python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.daily_tracking import DailyTracker, IntakeEntry, Profile, calculate_reference_targets

targets = calculate_reference_targets(Profile(30, 180, 80, "male", "active"))
tracker = DailyTracker(Path.home() / ".food-nutrition" / "intake.sqlite3")
tracker.upsert(IntakeEntry(
    entry_id="photo-2026-11-22-lunch",
    consumed_at=datetime.now(ZoneInfo("America/Chicago")),
    meal="lunch",
    nutrients={"energy_kcal": 600, "protein_g": 30, "carbohydrate_g": 80, "fat_g": 20},
    consumed_fraction=0.75,
    source="component-b",
))
print(tracker.daily_summary(datetime.now(ZoneInfo("America/Chicago")).date(), targets))
```

Run the dependency-free tests from the repository root:

```bash
python -m unittest tests.test_daily_tracking -v
```

The default Gordon endpoint is `https://wisc-housingdining.api.nutrislice.com`. Live menu availability remains an external dependency; automated tests use injected fixtures and do not contact it.
