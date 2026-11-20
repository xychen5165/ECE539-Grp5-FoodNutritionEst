"""Third development selector: plate geometry plus SAM-region appearance.

Assumes a light, approximately centered plate. Parameters are shared across
images; nutrition labels and dish IDs are never used for selection.
"""

import cv2
import numpy as np


def select_appearance(candidates, image):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    options = []
    for index, candidate in enumerate(candidates):
        mask = candidate['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        silhouette = np.zeros(mask.shape, dtype=np.uint8, order='C')
        cv2.drawContours(silhouette, [hull], -1, 1, cv2.FILLED)
        x, y, bw, bh = cv2.boundingRect(hull)
        fraction = float(silhouette.mean())
        center_distance = np.hypot((x + bw / 2) / width - .5, (y + bh / 2) / height - .5)
        compactness = float(silhouette.sum()) / (bw * bh)
        if not (.12 <= fraction <= .85 and .65 <= bw / bh <= 1.85
                and center_distance < .28 and compactness > .60):
            continue
        # A plate encloses a substantial light, low-saturation surface. This
        # avoids treating a large colorful meal cluster as the plate itself.
        neutral_fraction = float(((hsv[..., 1] < 60) & (hsv[..., 2] > 85))[mask > 0].mean())
        edge_contacts = sum([x < 5, y < 5, x + bw > width - 5, y + bh > height - 5])
        score = (2 * neutral_fraction + compactness + fraction
                 - center_distance - .35 * edge_contacts)
        options.append((score, index, silhouette.astype(bool), neutral_fraction))
    if not options:
        raise RuntimeError('No geometrically plausible light plate found')
    _, plate_index, silhouette, _ = max(options, key=lambda item: item[0])
    interior = cv2.erode(silhouette.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    plate_surface = candidates[plate_index]['segmentation'].astype(bool)
    neutral = interior & plate_surface & (hsv[..., 1] < 45)
    if neutral.sum() < 100:
        raise RuntimeError('Insufficient neutral plate surface for appearance reference')
    reference = np.median(lab[neutral], axis=0)
    # Food evidence: chromatic difference, saturation, or a marked darkening
    # relative to the local plate. Regions, rather than individual pixels,
    # allow pale food to be retained when SAM captures its whole outline.
    chroma_distance = np.linalg.norm(lab[..., 1:] - reference[1:], axis=-1)
    evidence = ((hsv[..., 1] > 55) | (chroma_distance > 14)
                | (lab[..., 0] < reference[0] - 35)) & interior
    union = np.zeros_like(interior)
    accepted = []
    for index, candidate in enumerate(candidates):
        if index == plate_index:
            continue
        mask = candidate['segmentation'].astype(bool)
        area = int(mask.sum())
        if area < 100 or area > .75 * interior.sum():
            continue
        containment = float((mask & interior).sum()) / area
        if containment < .85:
            continue
        contrast = float((mask & evidence).sum()) / area
        # Hole support retains white foods that differ weakly in color but
        # are excluded by SAM from the plate surface.
        hole_support = float((mask & interior & ~plate_surface).sum()) / area
        if contrast >= .40 or hole_support >= .65:
            union |= mask & interior
            accepted.append({'index': index, 'contrast_fraction': round(contrast, 4),
                             'hole_fraction': round(hole_support, 4)})
    # Preserve food holes even when no separate automatic candidate survived.
    holes = (interior & ~plate_surface).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 100 <= area <= .75 * interior.sum():
            union |= component
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(union)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 100:
            cleaned |= labels == label
    if not cleaned.any():
        raise RuntimeError('No food evidence within selected plate')
    return cleaned, {
        'method': 'plate_appearance_v1', 'plate_candidate_index': plate_index,
        'plate_area_ratio': round(float(silhouette.mean()), 6),
        'plate_reference_lab': reference.tolist(), 'accepted_candidates': accepted,
        'food_area_pixels': int(cleaned.sum()), 'food_area_ratio': float(cleaned.mean()),
    }
