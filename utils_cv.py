import cv2
import numpy as np
import math
from pathlib import Path

LOOKAHEAD_PX = 400


# ---------------------------------------------------
# GOLD COIN MASK
# ---------------------------------------------------
def coin_mask_gold(image_bgr: np.ndarray) -> np.ndarray:
    """Binary mask for 'gold' coins in HSV: bright, saturated yellow/orange."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    gold_lo1 = np.array([18, 130, 200])  # slightly higher S/V
    gold_hi1 = np.array([36, 255, 255])
    gold_lo2 = np.array([10, 150, 185])
    gold_hi2 = np.array([18, 255, 255])

    m1 = cv2.inRange(hsv, gold_lo1, gold_hi1)
    m2 = cv2.inRange(hsv, gold_lo2, gold_hi2)
    m = cv2.bitwise_or(m1, m2)

    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(m)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30 or area > 4000:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity > 0.5:
            cv2.drawContours(keep, [c], -1, 255, -1)
    return (keep > 0).astype(np.uint8)


# ---------------------------------------------------
# BUILD GROUND MASK (GREEN + BROWN - COINS)
# ---------------------------------------------------
def build_ground_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    green_lo = np.array([35, 40, 40])
    green_hi = np.array([85, 255, 255])
    mg = cv2.inRange(hsv, green_lo, green_hi)

    brown_lo = np.array([10, 40, 30])
    brown_hi = np.array([30, 190, 190])  # slightly lower V cap
    mb = cv2.inRange(hsv, brown_lo, brown_hi)

    mg[: h // 4, :] = 0
    mb[: h // 4, :] = 0

    mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    ground = cv2.bitwise_or(mg, mb)

    coins = coin_mask_gold(image_bgr)
    ground = cv2.bitwise_and(ground, cv2.bitwise_not(coins * 255))

    ground = cv2.morphologyEx(
        ground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ground > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(ground)
    min_area = max(500, int(0.001 * h * w))

    for lab in range(1, num_labels):
        x, y, bw, bh, area = stats[lab]
        if area >= min_area and (y + bh >= h - 1):
            keep[labels == lab] = 255

    if not keep.any() and num_labels > 1:
        lab = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep[labels == lab] = 255

    return (keep > 0).astype(np.uint8)


# ---------------------------------------------------
# RED JEEP ANGLE DETECTOR
# ---------------------------------------------------
def angle_from_red_contour(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 90, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 60])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
        hsv, lower_red2, upper_red2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None

    (cx, cy), (w, h), theta = cv2.minAreaRect(cnt)
    angle = theta + 90 if w < h else theta
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180
    return (float(angle), (int(cx), int(cy)), cnt)


# ---------------------------------------------------
# GROUND RAYCAST + FINDER
# ---------------------------------------------------
def vertical_distance_to_ground(image_bgr: np.ndarray, cx: int, cy: int):
    """Return (dist_pixels (float or None), (hit_x, hit_y) or None)."""
    ground = build_ground_mask(image_bgr)
    h, w = ground.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    hit_y = None
    for y in range(cy, h):
        if ground[y, cx] == 1:
            hit_y = y
            break

    if hit_y is None:
        return None, None
    return float(hit_y - cy), (cx, int(hit_y))


def find_ground_y_at_x(ground_mask: np.ndarray, x: int, y_start: int = 20):
    h, w = ground_mask.shape
    x = int(np.clip(x, 0, w - 1))
    y_start = int(np.clip(y_start, 0, h - 1))

    for y in range(y_start, h):
        if ground_mask[y, x] == 1:
            return y
    return None


# ---------------------------------------------------
# FRAME ANNOTATOR
# ---------------------------------------------------
def annotate_frame(image_bgr: np.ndarray):
    res = angle_from_red_contour(image_bgr)
    annotated = image_bgr.copy()
    jeep_angle = None
    height_px = None
    ground_slope = None

    if res is None:
        cv2.putText(
            annotated,
            "Red jeep not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None, None, None

    jeep_angle, (cx, cy), cnt = res
    h, w = image_bgr.shape[:2]

    # 1. Jeep Angle visualization
    cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
    rad = math.radians(jeep_angle)
    length = int(max(w, h) * 1.2)
    dx, dy = int(math.cos(rad) * length), int(math.sin(rad) * length)
    cv2.line(annotated, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 3)

    cv2.putText(
        annotated,
        f"Jeep Angle: {jeep_angle:.1f} deg",
        (cx + 10, max(30, cy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Jeep Angle: {jeep_angle:.1f} deg",
        (cx + 10, max(30, cy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Height to ground
    dist_px, hit = vertical_distance_to_ground(image_bgr, cx, cy)
    if hit is not None:
        cv2.line(annotated, (cx, cy), hit, (0, 255, 255), 2)
        cv2.circle(annotated, hit, 5, (0, 255, 255), -1)
        height_px = max(0.0, dist_px) if dist_px is not None else None
    cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), -1)

    label = f"Ground dist: {0.0 if height_px is None else height_px:.1f} px"
    cv2.putText(
        annotated,
        label,
        (cx + 10, min(h - 10, cy + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (cx + 10, min(h - 10, cy + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. Ground slope
    ground_mask = build_ground_mask(image_bgr)
    x2 = cx + LOOKAHEAD_PX
    y2 = find_ground_y_at_x(ground_mask, x2, y_start=cy)

    if y2 is not None:
        dx_slope = float(x2 - cx)
        dy_slope = float(cy - y2)
        ground_slope = math.degrees(math.atan2(dy_slope, dx_slope))

        p1 = (int(cx), int(cy))
        p2 = (int(x2), int(y2))
        p_corner = (int(x2), int(cy))

        cv2.line(annotated, p1, p2, (255, 0, 255), 3)
        cv2.line(annotated, p_corner, p2, (255, 100, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, p1, p_corner, (255, 100, 255), 1, cv2.LINE_AA)

        text_pos = (p2[0] + 5, p2[1])
        cv2.putText(
            annotated,
            f"Slope: {ground_slope:.1f} deg",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Slope: {ground_slope:.1f} deg",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, jeep_angle, height_px, ground_slope
