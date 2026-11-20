#!/usr/bin/env python3
"""Run frozen SAM ViT-B selectors on Nutrition5k overhead RGB images.

The candidate-union baseline favors stable, high-confidence masks near the
image center. The plate-complement alternative infers food from holes in a
filled plate silhouette.
Outputs are diagnostic only and must be reviewed by a person before they are
used as food masks in a model experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from appearance_selector import select_appearance


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def score_candidate(candidate: dict, height: int, width: int) -> dict:
    area_ratio = candidate["area"] / float(height * width)
    x, y, box_width, box_height = candidate["bbox"]
    mask_center_x = x + box_width / 2
    mask_center_y = y + box_height / 2
    image_center_x = width / 2
    image_center_y = height / 2
    diagonal = (width**2 + height**2) ** 0.5
    center_distance = (
        (mask_center_x - image_center_x) ** 2 + (mask_center_y - image_center_y) ** 2
    ) ** 0.5 / diagonal
    center_score = max(0.0, 1.0 - 2.0 * center_distance)
    center_inside = bool(candidate["segmentation"][height // 2, width // 2])

    eligible = 0.02 <= area_ratio <= 0.85
    score = (
        0.40 * float(candidate["predicted_iou"])
        + 0.35 * float(candidate["stability_score"])
        + 0.15 * center_score
        + 0.10 * float(center_inside)
    )
    return {
        "score": score,
        "eligible": eligible,
        "area_ratio": area_ratio,
        "center_distance_normalized": center_distance,
        "center_inside": center_inside,
    }


def centered_cluster_fallback(
    candidates: list[dict], height: int, width: int
) -> tuple[np.ndarray, dict]:
    """Use the largest plausible centered cluster when SAM has no plate mask."""
    image_area = float(height * width)
    cluster_options = []
    for index, candidate in enumerate(candidates):
        area_ratio = candidate["area"] / image_area
        x, y, box_width, box_height = candidate["bbox"]
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        if (
            0.01 <= area_ratio <= 0.15
            and 0.25 <= center_x <= 0.75
            and 0.20 <= center_y <= 0.70
            and box_width / max(1, box_height) <= 3.0
            and box_height / max(1, box_width) <= 3.0
        ):
            cluster_options.append((area_ratio, index, candidate))
    if not cluster_options:
        raise RuntimeError("No plate or centered food-cluster candidate was found")
    _, cluster_index, cluster = max(cluster_options)
    cluster_mask = cluster["segmentation"].astype(bool)
    return cluster_mask, {
        "method": "largest centered food-cluster candidate; no plate mask available",
        "food_candidate_indices": [cluster_index],
        "food_candidate_count": 1,
        "food_area_pixels": int(cluster_mask.sum()),
        "food_area_ratio": float(cluster_mask.sum()) / image_area,
    }


def find_plate(
    candidates: list[dict], height: int, width: int
) -> tuple[int, dict, np.ndarray] | None:
    """Return the best centered plate candidate and its filled outer silhouette."""
    image_area = float(height * width)
    plate_options = []
    for index, candidate in enumerate(candidates):
        area_ratio = candidate["area"] / image_area
        metrics = score_candidate(candidate, height, width)
        if 0.15 <= area_ratio <= 0.75 and metrics["center_distance_normalized"] <= 0.20:
            plate_options.append((metrics["score"], index, candidate))

    if not plate_options:
        return None
    _, plate_index, plate = max(plate_options)
    plate_mask = plate["segmentation"].astype(bool)
    plate_contours, _ = cv2.findContours(
        plate_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    plate_silhouette = np.zeros((height, width), dtype=np.uint8)
    if plate_contours:
        largest_contour = max(plate_contours, key=cv2.contourArea)
        cv2.drawContours(plate_silhouette, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return plate_index, plate, plate_silhouette.astype(bool)


def select_candidate_union(
    candidates: list[dict], height: int, width: int
) -> tuple[np.ndarray, dict]:
    """Baseline: union confident small SAM candidates inside a centered plate."""
    image_area = float(height * width)
    plate_context = find_plate(candidates, height, width)
    if plate_context is None:
        return centered_cluster_fallback(candidates, height, width)
    plate_index, plate, plate_silhouette = plate_context
    plate_x, plate_y, plate_width, plate_height = plate["bbox"]
    margin_x = 0.04 * plate_width
    margin_y = 0.04 * plate_height

    food_candidates = []
    for index, candidate in enumerate(candidates):
        if index == plate_index:
            continue
        mask = candidate["segmentation"].astype(bool)
        area = max(1, int(candidate["area"]))
        area_ratio = area / image_area
        x, y, box_width, box_height = candidate["bbox"]
        center_x = x + box_width / 2
        center_y = y + box_height / 2
        center_inside_plate_box = (
            plate_x + margin_x <= center_x <= plate_x + plate_width - margin_x
            and plate_y + margin_y <= center_y <= plate_y + plate_height - margin_y
        )
        inside_plate_ratio = float(np.logical_and(mask, plate_silhouette).sum()) / area
        if not 0.0015 <= area_ratio <= 0.12:
            continue
        if not center_inside_plate_box:
            continue
        if inside_plate_ratio < 0.90:
            continue
        if float(candidate["predicted_iou"]) < 0.90 or float(candidate["stability_score"]) < 0.92:
            continue
        food_candidates.append((index, candidate, inside_plate_ratio))

    if not food_candidates:
        raise RuntimeError("A plate was found, but no confident food candidates were inside it")

    union = np.zeros((height, width), dtype=bool)
    for _, candidate, _ in food_candidates:
        union |= candidate["segmentation"].astype(bool)
    return union, {
        "method": "candidate_union baseline",
        "plate_candidate_index": plate_index,
        "plate_area_ratio": float(plate["area"]) / image_area,
        "food_candidate_indices": [index for index, _, _ in food_candidates],
        "food_candidate_count": len(food_candidates),
        "food_area_pixels": int(union.sum()),
        "food_area_ratio": float(union.sum()) / image_area,
        "plate_bbox_xywh": [float(value) for value in plate["bbox"]],
        "minimum_inside_plate_ratio": min(ratio for _, _, ratio in food_candidates),
    }


def select_plate_complement(
    candidates: list[dict], height: int, width: int
) -> tuple[np.ndarray, dict]:
    """Method 2: infer food from holes inside a filled plate silhouette."""
    image_area = float(height * width)
    plate_context = find_plate(candidates, height, width)
    if plate_context is None:
        mask, details = centered_cluster_fallback(candidates, height, width)
        details["method"] = "plate_complement fallback: " + details["method"]
        return mask, details

    plate_index, plate, plate_silhouette = plate_context
    plate_mask = plate["segmentation"].astype(bool)
    kernel_size = max(5, int(round(min(height, width) * 0.025)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    plate_interior = cv2.erode(plate_silhouette.astype(np.uint8), kernel).astype(bool)
    raw_food = np.logical_and(plate_interior, np.logical_not(plate_mask)).astype(np.uint8)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_food = cv2.morphologyEx(raw_food, cv2.MORPH_OPEN, cleanup_kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_food, connectivity=8)
    minimum_component_area = max(100, int(round(image_area * 0.0005)))
    maximum_component_area = int(round(plate_silhouette.sum() * 0.80))
    food_mask = np.zeros((height, width), dtype=bool)
    kept_components = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if minimum_component_area <= area <= maximum_component_area:
            food_mask |= labels == label
            kept_components.append({"label": label, "area_pixels": area})

    if not kept_components:
        raise RuntimeError("Plate complement produced no food-sized components")
    return food_mask, {
        "method": "plate_complement",
        "plate_candidate_index": plate_index,
        "plate_area_ratio": float(plate["area"]) / image_area,
        "plate_bbox_xywh": [float(value) for value in plate["bbox"]],
        "erosion_kernel_pixels": kernel_size,
        "minimum_component_area": minimum_component_area,
        "kept_component_count": len(kept_components),
        "kept_components": kept_components,
        "food_area_pixels": int(food_mask.sum()),
        "food_area_ratio": float(food_mask.sum()) / image_area,
    }


def save_outputs(image: np.ndarray, mask: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = mask.astype(np.uint8)

    Image.fromarray(image).save(output_dir / "original.png")
    Image.fromarray(mask * 255, mode="L").save(output_dir / "mask.png")

    neutral = np.full_like(image, 127)
    masked = np.where(mask[..., None].astype(bool), image, neutral)
    Image.fromarray(masked).save(output_dir / "masked_rgb.png")

    overlay = image.astype(np.float32).copy()
    color = np.array([0, 220, 80], dtype=np.float32)
    overlay[mask.astype(bool)] = 0.55 * overlay[mask.astype(bool)] + 0.45 * color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 2)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(output_dir / "overlay.png")


def process_image(mask_generator, image_path: Path, output_root: Path, selector: str) -> dict:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    started = time.perf_counter()
    candidates = mask_generator.generate(image)
    elapsed = time.perf_counter() - started
    if not candidates:
        raise RuntimeError(f"SAM returned no masks for {image_path}")

    scored = []
    for index, candidate in enumerate(candidates):
        metrics = score_candidate(candidate, height, width)
        scored.append((metrics["score"], index, candidate, metrics))
    if selector == "candidate_union":
        food_mask, selection = select_candidate_union(candidates, height, width)
    elif selector == "plate_complement":
        food_mask, selection = select_plate_complement(candidates, height, width)
    elif selector == "plate_appearance":
        food_mask, selection = select_appearance(candidates, image)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    dish_id = image_path.parent.name
    output_dir = output_root / dish_id
    save_outputs(image, food_mask, output_dir)
    # A successful retry supersedes the failure marker for this same run folder.
    (output_dir / "failure.json").unlink(missing_ok=True)

    ranked = sorted(scored, reverse=True)[:10]
    result = {
        "dish_id": dish_id,
        "input_path": image_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "image_width": width,
        "image_height": height,
        "candidate_count": len(candidates),
        "elapsed_seconds": round(elapsed, 3),
        "selected": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in selection.items()
        },
        "top_candidates": [
            {
                "candidate_index": index,
                "score": round(score, 6),
                "area_ratio": round(metrics["area_ratio"], 6),
                "predicted_iou": round(float(candidate["predicted_iou"]), 6),
                "stability_score": round(float(candidate["stability_score"]), 6),
                "center_inside": metrics["center_inside"],
                "eligible": metrics["eligible"],
            }
            for score, index, candidate, metrics in ranked
        ],
    }
    (output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more RGB image paths")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sam_debug"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument(
        "--selector",
        choices=["candidate_union", "plate_complement", "plate_appearance"],
        default="candidate_union",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed image")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        parser.error("input images not found: " + ", ".join(missing))

    device = choose_device(args.device)
    print(f"Loading SAM ViT-B on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    results = []
    for image_path in args.inputs:
        print(f"Processing {image_path}...")
        try:
            result = process_image(mask_generator, image_path, args.output_dir, args.selector)
            result["status"] = "completed"
            results.append(result)
            print(
                f"  {result['candidate_count']} candidates; selected area "
                f"{result['selected']['food_area_ratio']:.1%}; {result['elapsed_seconds']:.1f}s"
            )
        except Exception as error:
            failure = {
                "dish_id": image_path.parent.name,
                "input_path": image_path.as_posix(),
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            results.append(failure)
            failure_dir = args.output_dir / image_path.parent.name
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"  FAILED: {type(error).__name__}: {error}")
            if args.fail_fast:
                raise

    summary = {
        "checkpoint": args.checkpoint.as_posix(),
        "device": device,
        "points_per_side": args.points_per_side,
        "selector": args.selector,
        "selector_status": "automatic first pass; manual review required",
        "completed_count": sum(result["status"] == "completed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "images": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
