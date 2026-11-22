# D - Gordon meal planning

Component D is implemented in [`src/meal_planner.py`](../src/meal_planner.py). It ranks Gordon menu combinations for review. A recommendation enters component C's intake log only after the user confirms consumption.

## What it does

- Enumerates combinations of one to three distinct menu items at 0.5, 1, or 1.5 listed servings.
- Rejects items without complete energy, protein, carbohydrate, and fat values. Nutrition must be expressed per listed serving.
- Applies confirmed exclusions against item names, stations, ingredients, allergens, and dietary tags. When exclusions are active, an item with no dietary metadata is excluded as uncertain.
- Enforces an optional ceiling on projected daily energy.
- Ranks the same feasible candidate set with a joint score and a protein-only baseline. The joint score equally weights normalized energy-target deviation, protein shortfall, carbohydrate distance from its planning band, and fat distance from its planning band. Band penalties are zero inside the band and normalized by its midpoint. Ties favor fewer foods.
- Returns portions, stations, nutrient totals, remaining target values, exclusion counts, warnings, and top-choice comparison. It reports `already_met` and `infeasible` states explicitly.
- Discloses incomplete daily logs; missing values are not treated as known zeroes without a warning.

## Python interface

```python
from src.meal_planner import plan_meals

result = plan_meals(
    menu_items=[
        {
            "id": "item-123",
            "name": "Example entree",
            "station": "Grill",
            "serving_size": "1 serving",
            "nutrition": {
                "energy_kcal": 420,
                "protein_g": 28,
                "carbohydrate_g": 36,
                "fat_g": 16,
            },
            "ingredients": ["chicken", "rice"],
            "allergens": [],
            "dietary_tags": ["gluten-free"],
        }
    ],
    daily_totals={
        "energy_kcal": 1500,
        "protein_g": 60,
        "carbohydrate_g": 170,
        "fat_g": 45,
    },
    targets={"energy_kcal": 2200, "protein_g": 90},
    carbohydrate_range_g=(248, 358),
    fat_range_g=(49, 86),
    exclusions=["peanut"],
    energy_ceiling_kcal=2200,
    top_k=5,
)
```

Menu items accept `id` (or `item_id`/`name`), optional `name`, `station`, and `serving_size`, and nutrient fields either at top level or nested in `nutrition`. The daily nutrient keys are `energy_kcal`, `protein_g`, `carbohydrate_g`, and `fat_g`. Pass `incomplete_log=True` if the upstream log is known to be incomplete even when all four totals are present. The planner uses only the Python standard library.

## Output and interpretation

`result["status"]` is `ok`, `already_met`, or `infeasible`. Ranked rows are in `result["rankings"]["joint"]` and `result["rankings"]["protein_only"]`; each row contains chosen items, meal nutrients, projected daily totals, remaining target values, and joint-score components. For macro ranges, remaining is positive below the range, zero inside it, and negative when above it. `result["comparison"]` reports whether the methods chose the same items and servings and the joint-score regret of the protein-only top choice. This is a ranking comparison over identical feasible candidates, not evidence of clinical benefit.

The component accepts the dated Gordon menu from C; it does not fetch menus or update the intake ledger. Do not commit personal profiles/logs, raw datasets, or private identity mappings.
