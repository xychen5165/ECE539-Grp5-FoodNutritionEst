"""Component C: reference targets, intake tracking, and Gordon menus."""

from .menu import GordonMenuClient, MenuDataError
from .targets import DailyTargets, Profile, calculate_reference_targets
from .tracker import DailyTracker, IntakeEntry

__all__ = [
    "DailyTargets",
    "DailyTracker",
    "GordonMenuClient",
    "IntakeEntry",
    "MenuDataError",
    "Profile",
    "calculate_reference_targets",
]
