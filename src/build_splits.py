"""Build training / development / test split files for Nutrition5k overhead photos.

Usage:
    python src/build_splits.py --data-root ../data/nutrition5k

Reads the Nutrition5k metadata files (no photos needed) and the calorie-lens plate groups, and writes:
    data/splits/train.csv, dev.csv, test.csv   one row per dish
    data/splits/dropped.csv                    removed dishes and the reason
    data/splits/plate_links.csv                every same-plate link and which method found it
    data/splits/summary.json                   counts, settings and file hashes

Rules (see data/splits/README.md for the reasoning):
    1. Cafe 1 dishes only (as in the Nutrition5k paper).
    2. Keep only dishes that have an overhead photo folder (overhead_dirs.txt).
    3. Drop noisy data: zero energy, mass under 5 g, mass over 2000 g, or a photo
       with no food (PHOTO_PROBLEMS).
    4. Group repeat shots of the same physical plate. Two dishes are linked when
       either method links them:
       a. Ingredient rule: a dish is linked to the dish recorded immediately
          before it when all of these hold:
            - they are at most 120 seconds apart;
            - every ingredient of the earlier dish is in the later dish with at
              least the same weight (tolerance: 5% or 1 g, whichever is larger);
            - the later dish's total mass is not lower than the earlier one's
              (tolerance: 5% or 1 g, whichever is larger).
       b. calorie-lens groups: dishes sharing a `group` in the calorie-lens
          dataset's dishes.parquet (https://huggingface.co/datasets/shalev396/calorie-lens-data,
          CC BY 4.0). These catch repeat photos taken more than 120 seconds apart.
       Linked dishes form one plate group.
    5. Test = the official Nutrition5k test list. Official training dishes are
       split by plate group into training (about 80%) and development (about
       20%). Official training dishes that share a plate group with a test dish
       are dropped so no plate appears in both training and test.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Column order in dish_metadata_cafe1.csv (the file has no header row).
TARGETS = ["energy_kcal", "mass_g", "fat_g", "carbohydrate_g", "protein_g"]
OUTPUT_TARGETS = ["mass_g", "energy_kcal", "protein_g", "carbohydrate_g", "fat_g"]

MAX_GAP_SECONDS = 120
WEIGHT_TOLERANCE_FRACTION = 0.05
WEIGHT_TOLERANCE_G = 1.0
MASS_TOLERANCE_FRACTION = 0.05
MASS_TOLERANCE_G = 1.0
MIN_MASS_G = 5.0
MAX_MASS_G = 2000.0
DEV_FRACTION = 0.2

# Dishes whose labels look normal but whose photo shows no food.
PHOTO_PROBLEMS = {"dish_1559172356": "photo shows a phone, not food"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_dishes(path):
    dishes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            dish_id = row[0]
            values = dict(zip(TARGETS, (float(x) for x in row[1:6])))
            ingredients = defaultdict(float)  # ingredient ID -> grams
            for i in range(6, len(row), 7):
                ingredients[row[i]] += float(row[i + 2])
            values["n_ingredients"] = len(ingredients)
            values["deprecated_ingredient"] = any(row[i + 1] == "deprecated" for i in range(6, len(row), 7))
            values["ingredients"] = dict(ingredients)
            values["time"] = int(dish_id.split("_")[1])
            dishes[dish_id] = values
    return dishes


def is_same_plate(a, b):
    """True when dish b (recorded right after a) looks like the same plate with food added or re-photographed."""
    if b["time"] - a["time"] > MAX_GAP_SECONDS:
        return False
    if b["mass_g"] < a["mass_g"] - max(MASS_TOLERANCE_FRACTION * a["mass_g"], MASS_TOLERANCE_G):
        return False
    for ingredient, grams in a["ingredients"].items():
        tolerance = max(WEIGHT_TOLERANCE_FRACTION * grams, WEIGHT_TOLERANCE_G)
        if b["ingredients"].get(ingredient, 0.0) < grams - tolerance:
            return False
    return True


def read_extra_groups(path):
    """Return dish_id -> group ID from the calorie-lens dishes.parquet file."""
    import pandas as pd  # only needed for this file

    table = pd.read_parquet(path, columns=["dish_id", "group"])
    return dict(zip(table["dish_id"], table["group"]))


def plate_groups(dishes, extra_groups=None):
    """Return dish_id -> group name, and every link as (earlier, later, source)."""
    ordered = sorted(dishes, key=lambda d: dishes[d]["time"])
    parent = {d: d for d in ordered}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    sources = defaultdict(set)  # (earlier, later) -> methods that link them
    for earlier, later in zip(ordered, ordered[1:]):
        if is_same_plate(dishes[earlier], dishes[later]):
            sources[(earlier, later)].add("ingredient rule")
    if extra_groups:
        members = defaultdict(list)
        for d in ordered:  # time order
            if d in extra_groups:
                members[extra_groups[d]].append(d)
        for ds in members.values():
            for earlier, later in zip(ds, ds[1:]):
                sources[(earlier, later)].add("calorie-lens")

    links = []
    for (earlier, later), methods in sorted(sources.items(), key=lambda kv: dishes[kv[0][0]]["time"]):
        parent[find(later)] = find(earlier)
        links.append((earlier, later, " + ".join(sorted(methods))))

    roots = {}
    groups = {}
    for d in ordered:
        r = find(d)
        if r not in roots:
            roots[r] = f"group_{len(roots) + 1:05d}"
        groups[d] = roots[r]
    return groups, links


def write_links(path, links, dishes, groups):
    """Every inferred same-plate link with its evidence, for manual review."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plate_group", "earlier_dish", "later_dish", "found_by", "gap_seconds",
                    "earlier_mass_g", "later_mass_g", "mass_change_pct",
                    "earlier_n_ingredients", "later_n_ingredients", "identical_ingredients"])
        for a_id, b_id, source in links:
            a, b = dishes[a_id], dishes[b_id]
            w.writerow([groups[a_id], a_id, b_id, source, b["time"] - a["time"],
                        a["mass_g"], b["mass_g"], round(100 * (b["mass_g"] - a["mass_g"]) / a["mass_g"], 1),
                        a["n_ingredients"], b["n_ingredients"],
                        int(set(a["ingredients"]) == set(b["ingredients"]))])


def label_problem(v):
    if v["energy_kcal"] <= 0:
        return "zero energy"
    if v["mass_g"] < MIN_MASS_G:
        return f"mass under {MIN_MASS_G:g} g"
    if v["mass_g"] > MAX_MASS_G:
        return f"mass over {MAX_MASS_G:g} g"
    return None


def energy_mismatch(v):
    """Flag (do not drop) dishes whose energy disagrees with 9*fat + 4*carb + 4*protein by over 30%."""
    estimate = 9 * v["fat_g"] + 4 * v["carbohydrate_g"] + 4 * v["protein_g"]
    return v["energy_kcal"] > 20 and abs(estimate - v["energy_kcal"]) / v["energy_kcal"] > 0.3


def write_split(path, ids, dishes, groups, split):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "split", "plate_group", "image_path", *OUTPUT_TARGETS,
                    "n_ingredients", "energy_mismatch_flag", "deprecated_ingredient_flag"])
        for d in sorted(ids):
            v = dishes[d]
            w.writerow([d, split, groups[d], f"imagery/realsense_overhead/{d}/rgb.png",
                        *(round(v[t], 4) for t in OUTPUT_TARGETS),
                        v["n_ingredients"], int(energy_mismatch(v)), int(v["deprecated_ingredient"])])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "data" / "nutrition5k"),
                   help="Folder holding metadata/, dish_ids/ and overhead_dirs.txt")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "splits"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-groups", default=str(REPO_ROOT.parent / "data" / "calorie-lens" / "dishes.parquet"),
                   help="calorie-lens dishes.parquet (see data/splits/README.md for the download)")
    p.add_argument("--no-extra-groups", action="store_true", help="use only the ingredient rule")
    args = p.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dishes = read_dishes(root / "metadata" / "dish_metadata_cafe1.csv")
    overhead = set(read_lines(root / "overhead_dirs.txt"))
    official_train = set(read_lines(root / "dish_ids" / "splits" / "rgb_train_ids.txt"))
    official_test = set(read_lines(root / "dish_ids" / "splits" / "rgb_test_ids.txt"))

    extra_groups = None
    if not args.no_extra_groups:
        if not Path(args.extra_groups).exists():
            raise SystemExit(f"Missing {args.extra_groups}. Download it (see data/splits/README.md) "
                             "or run with --no-extra-groups.")
        extra_groups = read_extra_groups(args.extra_groups)

    # Group using every Cafe 1 dish so chains are not broken by dishes without photos.
    groups, links = plate_groups(dishes, extra_groups)
    write_links(out / "plate_links.csv", links, dishes, groups)

    dropped = []  # (dish_id, reason)
    kept = []
    for d in sorted(overhead):
        if d not in dishes:
            continue  # Cafe 2 dish or no labels; not counted as dropped
        if d not in official_train and d not in official_test:
            dropped.append((d, "not in an official split list"))
            continue
        problem = PHOTO_PROBLEMS.get(d) or label_problem(dishes[d])
        if problem:
            dropped.append((d, problem))
            continue
        kept.append(d)

    test = [d for d in kept if d in official_test]
    test_groups = {groups[d] for d in test}

    train_pool = []
    for d in kept:
        if d in official_test:
            continue
        if groups[d] in test_groups:
            dropped.append((d, "same plate group as a test dish"))
        else:
            train_pool.append(d)

    # Split the training pool by plate group.
    pool_groups = defaultdict(list)
    for d in train_pool:
        pool_groups[groups[d]].append(d)
    group_names = sorted(pool_groups)
    random.Random(args.seed).shuffle(group_names)
    dev_target = round(DEV_FRACTION * len(train_pool))
    dev, train = [], []
    for g in group_names:
        (dev if len(dev) < dev_target else train).extend(pool_groups[g])

    files = {"train": (train, out / "train.csv"), "dev": (dev, out / "dev.csv"), "test": (test, out / "test.csv")}
    for split, (ids, path) in files.items():
        write_split(path, ids, dishes, groups, split)

    with open(out / "dropped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "reason"])
        w.writerows(sorted(dropped))

    # Sanity checks: no dish or plate group in two splits.
    sets = {s: set(ids) for s, (ids, _) in files.items()}
    gsets = {s: {groups[d] for d in ids} for s, (ids, _) in files.items()}
    for a in sets:
        for b in sets:
            if a < b:
                assert not sets[a] & sets[b], f"dish overlap {a}/{b}"
                assert not gsets[a] & gsets[b], f"plate group overlap {a}/{b}"

    group_sizes = Counter(Counter(groups[d] for d in kept).values())
    summary = {
        "source": "Nutrition5k Cafe 1 metadata + overhead_dirs.txt + official rgb split lists",
        "extra_plate_groups": None if extra_groups is None else
            f"calorie-lens dishes.parquet (sha256 {file_hash(args.extra_groups)})",
        "seed": args.seed,
        "rules": {
            "max_gap_seconds_for_plate_link": MAX_GAP_SECONDS,
            "ingredient_weight_tolerance": f"{WEIGHT_TOLERANCE_FRACTION:.0%} or {WEIGHT_TOLERANCE_G:g} g",
            "total_mass_drop_tolerance": f"{MASS_TOLERANCE_FRACTION:.0%} or {MASS_TOLERANCE_G:g} g",
            "min_mass_g": MIN_MASS_G,
            "max_mass_g": MAX_MASS_G,
            "dev_fraction_of_official_train": DEV_FRACTION,
        },
        "counts": {
            "cafe1_dishes_with_labels": len(dishes),
            "cafe1_dishes_with_overhead_photo": sum(d in dishes for d in overhead),
            "kept_after_label_checks": len(kept),
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "dropped": dict(Counter(r for _, r in dropped)),
            "plate_groups": {s: len(g) for s, g in gsets.items()},
            "plate_links_by_method": dict(Counter(source for _, _, source in links)),
            "deprecated_ingredient_flagged": {s: sum(dishes[d]["deprecated_ingredient"] for d in ids)
                                              for s, (ids, _) in files.items()},
            "energy_mismatch_flagged": {s: sum(energy_mismatch(dishes[d]) for d in ids)
                                        for s, (ids, _) in files.items()},
        },
        "plate_group_size_distribution_kept": {str(k): v for k, v in sorted(group_sizes.items())},
        "sha256_first16": {p.name: file_hash(p) for _, p in files.values()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print("group sizes:", summary["plate_group_size_distribution_kept"])


if __name__ == "__main__":
    main()
