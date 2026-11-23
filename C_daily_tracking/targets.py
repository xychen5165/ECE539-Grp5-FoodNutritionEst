"""Editable reference nutrition targets for generally healthy adults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


_EER_COEFFICIENTS = {
    "male": {
        "inactive": (753.07, -10.83, 6.50, 14.10),
        "low_active": (581.47, -10.83, 8.30, 14.94),
        "active": (1004.82, -10.83, 6.52, 15.91),
        "very_active": (-517.88, -10.83, 15.61, 19.11),
    },
    "female": {
        "inactive": (584.90, -7.01, 5.72, 11.71),
        "low_active": (575.77, -7.01, 6.60, 12.14),
        "active": (710.25, -7.01, 6.54, 12.34),
        "very_active": (511.83, -7.01, 9.07, 12.56),
    },
}


@dataclass(frozen=True)
class Profile:
    """Inputs supported by the adult 2023 DRI EER equations.

    ``sex`` is the equation category, not gender identity. Pregnancy and
    breastfeeding require different equations and are intentionally unsupported.
    """

    age_years: float
    height_cm: float
    weight_kg: float
    sex: str
    activity: str

    def __post_init__(self) -> None:
        if self.age_years < 19:
            raise ValueError("adult reference targets require age_years >= 19")
        if self.height_cm <= 0 or self.weight_kg <= 0:
            raise ValueError("height_cm and weight_kg must be positive")
        if self.sex not in _EER_COEFFICIENTS:
            raise ValueError("sex must be 'male' or 'female' for the EER equation")
        if self.activity not in _EER_COEFFICIENTS[self.sex]:
            raise ValueError(
                "activity must be inactive, low_active, active, or very_active"
            )


@dataclass(frozen=True)
class DailyTargets:
    """Confirmed daily targets consumed by tracking and meal planning."""

    energy_kcal: float
    protein_g: float
    carbohydrate_range_g: tuple[float, float]
    fat_range_g: tuple[float, float]
    source: str = "2023 DRI adult EER; adult AMDR; protein RDA"

    def __post_init__(self) -> None:
        if self.energy_kcal <= 0 or self.protein_g <= 0:
            raise ValueError("energy and protein targets must be positive")
        for name, bounds in (
            ("carbohydrate_range_g", self.carbohydrate_range_g),
            ("fat_range_g", self.fat_range_g),
        ):
            if len(bounds) != 2 or bounds[0] < 0 or bounds[0] > bounds[1]:
                raise ValueError(f"invalid {name}")

    def with_overrides(self, **overrides: object) -> "DailyTargets":
        """Return a validated, user-confirmed copy with selected fields changed."""

        allowed = {
            "energy_kcal",
            "protein_g",
            "carbohydrate_range_g",
            "fat_range_g",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unsupported target override(s): {sorted(unknown)}")
        return replace(self, **overrides, source=f"{self.source}; user-confirmed edits")

    def as_dict(self) -> dict[str, object]:
        return {
            "energy_kcal": self.energy_kcal,
            "protein_g": self.protein_g,
            "carbohydrate_range_g": list(self.carbohydrate_range_g),
            "fat_range_g": list(self.fat_range_g),
            "source": self.source,
        }


def calculate_reference_targets(
    profile: Profile,
    *,
    energy_adjustment_kcal: float = 0.0,
    protein_g_per_kg: float = 0.8,
    overrides: Mapping[str, object] | None = None,
) -> DailyTargets:
    """Calculate editable starting targets for a generally healthy adult.

    Carbohydrate uses 45--65% and fat 20--35% of energy. Protein and
    carbohydrate use 4 kcal/g; fat uses 9 kcal/g. The energy adjustment and
    protein factor must be explicitly confirmed by the caller/user.
    """

    if protein_g_per_kg <= 0:
        raise ValueError("protein_g_per_kg must be positive")
    intercept, age_c, height_c, weight_c = _EER_COEFFICIENTS[profile.sex][
        profile.activity
    ]
    energy = (
        intercept
        + age_c * profile.age_years
        + height_c * profile.height_cm
        + weight_c * profile.weight_kg
        + energy_adjustment_kcal
    )
    if energy <= 0:
        raise ValueError("adjusted energy target must be positive")
    targets = DailyTargets(
        energy_kcal=round(energy, 2),
        protein_g=round(protein_g_per_kg * profile.weight_kg, 2),
        carbohydrate_range_g=(round(energy * 0.45 / 4, 2), round(energy * 0.65 / 4, 2)),
        fat_range_g=(round(energy * 0.20 / 9, 2), round(energy * 0.35 / 9, 2)),
    )
    return targets.with_overrides(**dict(overrides)) if overrides else targets
