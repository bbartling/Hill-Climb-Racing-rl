
import cv2
import numpy as np
import math
from pathlib import Path

# ------------------------------
# Red jeep angle detector
# ------------------------------
def angle_from_red_contour(image_bgr: np.ndarray):
    """Return (angle_degrees, (cx, cy), contour) or None if not found."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 90, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 60])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None

    (cx, cy), (w, h), theta = cv2.minAreaRect(cnt)
    angle = theta + 90 if w < h else theta
    # Normalize to [-90, 90]
    if angle < -90: angle += 180
    if angle > 90:  angle -= 180
    return (float(angle), (int(cx), int(cy)), cnt)

# ------------------------------
# Ground masking + vertical raycast (straight down)
# ------------------------------
def _green_brown_masks(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    green_lo = np.array([35, 40, 40])
    green_hi = np.array([85,255,255])
    brown_lo = np.array([10, 60, 40])
    brown_hi = np.array([30,255,255])

    mg = cv2.inRange(hsv, green_lo, green_hi)
    mb = cv2.inRange(hsv, brown_lo, brown_hi)

    h, w = mg.shape
    mg[:h//4, :] = 0
    mb[:h//4, :] = 0

    mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    return (mg > 0).astype(np.uint8), (mb > 0).astype(np.uint8)

def vertical_distance_to_ground(image_bgr: np.ndarray, cx: int, cy: int):
    """Return (dist_pixels (float or None), (hit_x, hit_y) or None)."""
    mask_green, mask_brown = _green_brown_masks(image_bgr)
    ground = (mask_green | mask_brown).astype(np.uint8)

    h, w = ground.shape
    cx = int(np.clip(cx, 0, w-1))
    cy = int(np.clip(cy, 0, h-1))

    hit_y = None
    for y in range(cy, h):     # ONLY downward
        if ground[y, cx] == 1:
            if mask_green[y, cx] == 1:
                y2 = y
                while y2 + 1 < h and mask_green[y2+1, cx] == 1:
                    y2 += 1
                hit_y = y2
            else:
                hit_y = y
            break

    if hit_y is None:
        return None, None
    return float(hit_y - cy), (cx, int(hit_y))

# ------------------------------
# One-frame annotator (used by images, video, record modes)
# ------------------------------
def annotate_frame(image_bgr: np.ndarray):
    """Return (annotated_bgr, angle_deg or None, height_px or None)."""
    res = angle_from_red_contour(image_bgr)
    annotated = image_bgr.copy()
    angle = None
    height_px = None

    if res is None:
        cv2.putText(annotated, "Red jeep not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        return annotated, angle, height_px

    angle, (cx, cy), cnt = res
    h, w = image_bgr.shape[:2]

    # Jeep contour
    cv2.drawContours(annotated, [cnt], -1, (0,255,0), 2)

    # Red infinite direction line
    rad = math.radians(angle)
    length = int(max(w, h) * 1.2)
    dx, dy = int(math.cos(rad)*length), int(math.sin(rad)*length)
    cv2.line(annotated, (cx-dx, cy-dy), (cx+dx, cy+dy), (0,0,255), 3)

    # Angle label
    cv2.putText(annotated, f"Angle: {angle:.1f} deg", (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(annotated, f"Angle: {angle:.1f} deg", (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Vertical ground distance
    dist_px, hit = vertical_distance_to_ground(image_bgr, cx, cy)
    if hit is not None:
        cv2.line(annotated, (cx, cy), hit, (0, 255, 255), 2)
        cv2.circle(annotated, hit, 5, (0, 255, 255), -1)
        height_px = max(0.0, dist_px) if dist_px is not None else None
    cv2.circle(annotated, (cx, cy), 6, (255,255,255), -1)

    # Height label
    label = f"Ground dist: {0.0 if height_px is None else height_px:.1f} px"
    cv2.putText(annotated, label, (cx + 10, min(h - 10, cy + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(annotated, label, (cx + 10, min(h - 10, cy + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return annotated, angle, height_px
