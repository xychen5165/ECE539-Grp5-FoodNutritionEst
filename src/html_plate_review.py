"""Build a side-by-side review page for checking inferred same-plate links by eye.

Usage:
    python src/html_plate_review.py --data-root ../data/nutrition5k

Writes data/splits/plate_pair_review.html. Photos load from Nutrition5k's public
storage, so the page works on any computer with internet access and no images
are stored in the repository.

Sections (pairs chosen with a fixed seed, so the page is reproducible):
    A. Linked pairs 61-120 seconds apart (grouped under the current rule).
    B. Pairs 121-600 seconds apart that pass the ingredient-weight and mass
       checks but are not linked because of the time cutoff.
    C. Two pairs with identical labels in different splits, about 20 minutes apart.
"""

import argparse
import csv
import html
import random
from collections import defaultdict
from pathlib import Path

from build_splits import (MASS_TOLERANCE_FRACTION, MASS_TOLERANCE_G, MAX_GAP_SECONDS,
                          REPO_ROOT, WEIGHT_TOLERANCE_FRACTION, WEIGHT_TOLERANCE_G)

PHOTO_URL = ("https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
             "imagery/realsense_overhead/{}/rgb.png")
PAIRS_PER_SECTION = 10
LONGEST_GAP_REVIEWED = 600
IDENTICAL_LABEL_PAIRS = [("dish_1559242066", "dish_1559243139"),
                         ("dish_1562689601", "dish_1562690886")]


def read_dishes(path):
    dishes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            by_id, names = defaultdict(float), defaultdict(float)
            for i in range(6, len(row), 7):
                by_id[row[i]] += float(row[i + 2])
                names[row[i + 1]] += float(row[i + 2])
            dishes[row[0]] = dict(time=int(row[0].split("_")[1]), energy=float(row[1]),
                                  mass=float(row[2]), by_id=dict(by_id), names=dict(names))
    return dishes


def passes_weight_and_mass(a, b):
    if b["mass"] < a["mass"] - max(MASS_TOLERANCE_FRACTION * a["mass"], MASS_TOLERANCE_G):
        return False
    return all(b["by_id"].get(k, 0.0) >= g - max(WEIGHT_TOLERANCE_FRACTION * g, WEIGHT_TOLERANCE_G)
               for k, g in a["by_id"].items())


def card(dish_id, d, split):
    ingredients = ", ".join(f"{html.escape(k)} {v:.0f} g"
                            for k, v in sorted(d["names"].items(), key=lambda kv: -kv[1]))
    return (f'<figure><img src="{PHOTO_URL.format(dish_id)}" alt="{dish_id}" loading="lazy">'
            f'<figcaption><b>{dish_id}</b> · {split}<br>{d["mass"]:.0f} g · {d["energy"]:.0f} kcal'
            f'<br><small>{ingredients}</small></figcaption></figure>')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "data" / "nutrition5k"))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "splits" / "plate_pair_review.html"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    root = Path(args.data_root)

    dishes = read_dishes(root / "metadata" / "dish_metadata_cafe1.csv")
    with open(root / "overhead_dirs.txt", encoding="utf-8") as f:
        overhead = {line.strip() for line in f if line.strip()}
    split_of = {}
    for split in ("train", "dev", "test"):
        with open(REPO_ROOT / "data" / "splits" / f"{split}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split_of[row["dish_id"]] = split

    ordered = sorted(dishes, key=lambda d: dishes[d]["time"])
    linked, cutoff_excluded = [], []
    for a, b in zip(ordered, ordered[1:]):
        if a not in overhead or b not in overhead or not passes_weight_and_mass(dishes[a], dishes[b]):
            continue
        gap = dishes[b]["time"] - dishes[a]["time"]
        if 60 < gap <= MAX_GAP_SECONDS:
            linked.append((a, b))
        elif MAX_GAP_SECONDS < gap <= LONGEST_GAP_REVIEWED:
            cutoff_excluded.append((a, b))

    rng = random.Random(args.seed)
    sections = [
        (f"A. Linked, 61–{MAX_GAP_SECONDS} seconds apart (currently grouped)",
         rng.sample(linked, min(PAIRS_PER_SECTION, len(linked)))),
        (f"B. Pass the weight and mass checks but not linked, "
         f"{MAX_GAP_SECONDS + 1}–{LONGEST_GAP_REVIEWED} seconds apart (outside the time cutoff)",
         rng.sample(cutoff_excluded, min(PAIRS_PER_SECTION, len(cutoff_excluded)))),
        ("C. Identical labels in different splits, about 20 minutes apart (not grouped)",
         IDENTICAL_LABEL_PAIRS),
    ]

    parts = [
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Plate pair review</title><style>"
        "body{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;background:#fafafa;color:#222}"
        ".pair{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:260px}"
        "img{width:100%;border-radius:4px;background:#eee}small{color:#666}h2{margin-top:32px}"
        "label{margin-right:16px}</style>",
        "<h1>Plate pair review</h1>",
        "<p>For each pair, decide whether both photos show the same physical plate: the food from the "
        "left photo is still present in the right one (food may be added). Use individual food pieces "
        "and marks on the plate; the plate's position can shift. Choices are not saved; record results "
        "in <code>docs/development-log.md</code>.</p>",
        f"<p>Current rule: at most {MAX_GAP_SECONDS} seconds apart, every earlier ingredient present "
        f"with at least its weight (tolerance {WEIGHT_TOLERANCE_FRACTION:.0%} or {WEIGHT_TOLERANCE_G:g} g), "
        f"total mass not dropping by more than {MASS_TOLERANCE_FRACTION:.0%} or {MASS_TOLERANCE_G:g} g. "
        f"Generated by <code>src/html_plate_review.py</code> (seed {args.seed}).</p>",
    ]
    n = 0
    for title, pairs in sections:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        for a, b in pairs:
            n += 1
            gap = dishes[b]["time"] - dishes[a]["time"]
            parts.append(
                f'<div class=pair><b>Pair {n}</b> — {gap} seconds apart'
                f'<div class=row>{card(a, dishes[a], split_of.get(a, "not in splits"))}'
                f'{card(b, dishes[b], split_of.get(b, "not in splits"))}</div>'
                + "".join(f'<label><input type=radio name=p{n}> {c}</label>'
                          for c in ("Same plate", "Different", "Unsure"))
                + "</div>")
    parts.append("<p><small>Photos and labels: Nutrition5k dataset (Thames et al., CVPR 2021), "
                 "Google Research, https://github.com/google-research-datasets/Nutrition5k. "
                 "Used under the dataset's licence; see its repository for terms.</small></p></html>")

    out = Path(args.out)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"{n} pairs written to {out}")


if __name__ == "__main__":
    main()
