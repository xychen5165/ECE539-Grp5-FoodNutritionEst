"""Constraint-aware Gordon menu planner for project component D.

The planner is deterministic and uses only the Python standard library. Menu
nutrition is interpreted per listed serving; missing values are never treated
as zero.
"""

from __future__ import annotations

from itertools import combinations, product
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


NUTRIENTS = ("energy_kcal", "protein_g", "carbohydrate_g", "fat_g")
PORTION_MULTIPLIERS = (0.5, 1.0, 1.5)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) and result >= 0 else None


def _nutrition(item: Mapping[str, Any]) -> dict[str, float] | None:
    raw = item.get("nutrition", item)
    if not isinstance(raw, Mapping):
        return None
    values = {key: _number(raw.get(key)) for key in NUTRIENTS}
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items()}


def _range(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    low, high = _number(value[0]), _number(value[1])
    if low is None or high is None or high < low:
        return None
    return low, high


def _item_id(item: Mapping[str, Any], index: int) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("name") or f"menu-item-{index}")


def _excluded(item: Mapping[str, Any], exclusions: set[str]) -> tuple[bool, bool]:
    """Return (excluded, dietary_metadata_uncertain)."""
    if not exclusions:
        return False, False
    fields: list[str] = []
    for key in ("name", "category", "station", "description"):
        value = item.get(key)
        if value:
            fields.append(str(value))
    metadata_present = False
    for key in ("ingredients", "allergens", "tags", "dietary_tags"):
        value = item.get(key)
        if value is not None:
            metadata_present = True
            fields.extend(str(part) for part in (value if isinstance(value, (list, tuple, set)) else [value]))
    searchable = " ".join(fields).casefold()
    if not metadata_present:
        # Do not claim an item is safe for a confirmed exclusion when the menu
        # carries no ingredient/allergen/dietary metadata to check.
        return True, True
    return any(token in searchable for token in exclusions), False


def _distance_to_range(value: float, bounds: tuple[float, float] | None) -> float:
    if bounds is None:
        return 0.0
    low, high = bounds
    gap = max(low - value, 0.0, value - high)
    midpoint = (low + high) / 2
    return gap / midpoint if midpoint > 0 else gap


def _remaining_range(value: float, bounds: tuple[float, float] | None) -> float | None:
    if bounds is None:
        return None
    low, high = bounds
    if value < low:
        return low - value
    if value > high:
        return high - value
    return 0.0


def _score(
    totals: Mapping[str, float],
    targets: Mapping[str, float],
    carb_range: tuple[float, float] | None,
    fat_range: tuple[float, float] | None,
) -> tuple[float, dict[str, float]]:
    energy_target = targets.get("energy_kcal", 0.0)
    protein_target = targets.get("protein_g", 0.0)
    penalties = {
        "energy_deviation": abs(totals["energy_kcal"] - energy_target) / energy_target if energy_target > 0 else 0.0,
        "protein_shortfall": max(protein_target - totals["protein_g"], 0.0) / protein_target if protein_target > 0 else 0.0,
        "carbohydrate_range_distance": _distance_to_range(totals["carbohydrate_g"], carb_range),
        "fat_range_distance": _distance_to_range(totals["fat_g"], fat_range),
    }
    return sum(penalties.values()) / len(penalties), penalties


def _goal_state(totals: Mapping[str, float], targets: Mapping[str, float], carb_range: tuple[float, float] | None, fat_range: tuple[float, float] | None) -> bool:
    return (
        totals["energy_kcal"] >= targets.get("energy_kcal", float("inf"))
        and totals["protein_g"] >= targets.get("protein_g", float("inf"))
        and (carb_range is None or carb_range[0] <= totals["carbohydrate_g"] <= carb_range[1])
        and (fat_range is None or fat_range[0] <= totals["fat_g"] <= fat_range[1])
    )


def plan_meals(
    menu_items: Iterable[Mapping[str, Any]],
    daily_totals: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    exclusions: Iterable[str] = (),
    energy_ceiling_kcal: float | None = None,
    carbohydrate_range_g: Sequence[float] | None = None,
    fat_range_g: Sequence[float] | None = None,
    incomplete_log: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Rank 1–3 food combinations under common constraints.

    Each item requires an id or name, and complete nutrition per listed serving
    under energy_kcal, protein_g, carbohydrate_g and fat_g (at top level or in
    a ``nutrition`` mapping). Optional dietary metadata uses ingredients,
    allergens, tags or dietary_tags. Exclusions are case-insensitive substrings.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    clean_targets = {key: _number(targets.get(key)) for key in NUTRIENTS}
    if clean_targets["energy_kcal"] is None or clean_targets["protein_g"] is None:
        raise ValueError("targets must include nonnegative energy_kcal and protein_g")
    target_values = {key: float(value) for key, value in clean_targets.items() if value is not None}
    parsed_daily = {key: _number(daily_totals.get(key)) for key in NUTRIENTS}
    incomplete_log = incomplete_log or any(value is None for value in parsed_daily.values())
    current = {key: value or 0.0 for key, value in parsed_daily.items()}
    carb_range = _range(carbohydrate_range_g)
    fat_range = _range(fat_range_g)
    if carbohydrate_range_g is not None and carb_range is None:
        raise ValueError("carbohydrate_range_g must be a [minimum, maximum] pair")
    if fat_range_g is not None and fat_range is None:
        raise ValueError("fat_range_g must be a [minimum, maximum] pair")
    ceiling = _number(energy_ceiling_kcal) if energy_ceiling_kcal is not None else None
    if energy_ceiling_kcal is not None and ceiling is None:
        raise ValueError("energy_ceiling_kcal must be nonnegative")
    excluded_tokens = {str(value).strip().casefold() for value in exclusions if str(value).strip()}

    eligible: list[tuple[str, Mapping[str, Any], dict[str, float]]] = []
    counts = {"incomplete_nutrition": 0, "dietary_exclusion": 0, "uncertain_dietary_metadata": 0, "energy_ceiling": 0}
    seen_ids: set[str] = set()
    for index, item in enumerate(menu_items):
        if not isinstance(item, Mapping):
            counts["incomplete_nutrition"] += 1
            continue
        identifier = _item_id(item, index)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        nutrition = _nutrition(item)
        if nutrition is None:
            counts["incomplete_nutrition"] += 1
            continue
        is_excluded, uncertain = _excluded(item, excluded_tokens)
        if is_excluded:
            counts["uncertain_dietary_metadata" if uncertain else "dietary_exclusion"] += 1
            continue
        eligible.append((identifier, item, nutrition))

    already_met = _goal_state(current, target_values, carb_range, fat_range)
    candidates: list[dict[str, Any]] = []
    if not already_met:
        for size in range(1, min(3, len(eligible)) + 1):
            for group in combinations(eligible, size):
                for portions in product(PORTION_MULTIPLIERS, repeat=size):
                    meal_nutrients = {key: sum(food[2][key] * portion for food, portion in zip(group, portions)) for key in NUTRIENTS}
                    projected = {key: current[key] + meal_nutrients[key] for key in NUTRIENTS}
                    if ceiling is not None and projected["energy_kcal"] > ceiling:
                        counts["energy_ceiling"] += 1
                        continue
                    composite_score, penalties = _score(projected, target_values, carb_range, fat_range)
                    protein_shortfall = max(target_values["protein_g"] - projected["protein_g"], 0.0)
                    selections = []
                    for (identifier, item, _), portion in zip(group, portions):
                        selections.append({
                            "item_id": identifier,
                            "name": str(item.get("name") or identifier),
                            "station": item.get("station"),
                            "servings": portion,
                            "listed_serving": item.get("serving_size") or item.get("serving"),
                        })
                    candidates.append({
                        "items": selections,
                        "meal_nutrients": meal_nutrients,
                        "projected_daily_totals": projected,
                        "remaining": {
                            "energy_kcal": target_values["energy_kcal"] - projected["energy_kcal"],
                            "protein_g": target_values["protein_g"] - projected["protein_g"],
                            "carbohydrate_g": target_values.get("carbohydrate_g", _remaining_range(projected["carbohydrate_g"], carb_range)),
                            "fat_g": target_values.get("fat_g", _remaining_range(projected["fat_g"], fat_range)),
                        },
                        "joint_score": composite_score,
                        "joint_penalties": penalties,
                        "protein_only_score": protein_shortfall / target_values["protein_g"] if target_values["protein_g"] > 0 else 0.0,
                        "constraint_violations": [],
                    })

    def stable_key(candidate: Mapping[str, Any], score_key: str) -> tuple[Any, ...]:
        return (
            candidate[score_key],
            len(candidate["items"]),
            sum(item["servings"] for item in candidate["items"]),
            tuple(item["item_id"] for item in candidate["items"]),
            tuple(item["servings"] for item in candidate["items"]),
        )

    joint = sorted(candidates, key=lambda candidate: stable_key(candidate, "joint_score"))
    protein_only = sorted(candidates, key=lambda candidate: stable_key(candidate, "protein_only_score"))
    if already_met:
        status = "already_met"
    elif not joint:
        status = "infeasible"
    else:
        status = "ok"

    def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[:top_k]

    warnings = ["Daily totals may be incomplete; recommendations use only the logged intake supplied."] if incomplete_log else []
    if excluded_tokens:
        warnings.append("Items without dietary metadata were excluded while confirmed exclusions were active.")
    return {
        "status": status,
        "message": {
            "ok": "Ranked meals satisfy the supplied hard constraints.",
            "already_met": "Logged intake already meets the supplied energy, protein, and macro-range goals.",
            "infeasible": "No complete menu combination satisfies the supplied dietary and energy constraints.",
        }[status],
        "constraints": {
            "max_foods": 3,
            "portion_multipliers": list(PORTION_MULTIPLIERS),
            "exclusions": sorted(excluded_tokens),
            "energy_ceiling_kcal": ceiling,
            "carbohydrate_range_g": list(carb_range) if carb_range else None,
            "fat_range_g": list(fat_range) if fat_range else None,
        },
        "counts": {**counts, "eligible_foods": len(eligible), "feasible_candidates": len(candidates)},
        "warnings": warnings,
        "rankings": {
            "joint": public_rows(joint),
            "protein_only": public_rows(protein_only),
        },
        "comparison": compare_rankings(joint, protein_only),
    }


def compare_rankings(joint: Sequence[Mapping[str, Any]], protein_only: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize top-choice differences using the same feasible candidate set."""
    if not joint or not protein_only:
        return {"same_feasible_set": len(joint) == len(protein_only), "top_choice_item_ids_match": None, "joint_top_score": None, "protein_only_top_joint_score": None}
    joint_top = joint[0]
    protein_top = protein_only[0]
    ids = lambda row: tuple((item["item_id"], item["servings"]) for item in row["items"])
    return {
        "same_feasible_set": len(joint) == len(protein_only),
        "top_choice_match": ids(joint_top) == ids(protein_top),
        "joint_top_score": joint_top["joint_score"],
        "protein_only_top_joint_score": protein_top["joint_score"],
        "joint_top_items": ids(joint_top),
        "protein_only_top_items": ids(protein_top),
        "joint_score_regret_of_protein_only": protein_top["joint_score"] - joint_top["joint_score"],
    }
