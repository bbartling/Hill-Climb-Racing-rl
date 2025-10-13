# jeep_angle_testing.py
# Angle + direction line (existing), now with STRAIGHT-DOWN ground distance (pixels).
import cv2, numpy as np, math
from pathlib import Path

# ---------- Existing: angle from red jeep contour ----------
def angle_from_red_contour(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red mask (jeep body)
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
    if angle < -90: angle += 180
    if angle > 90:  angle -= 180
    return (float(angle), (int(cx), int(cy)), cnt)

# ---------- Ground masking ----------
def _green_brown_masks(image_bgr: np.ndarray):
    """Return (mask_green, mask_brown) as uint8 0/1 masks."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Grass (bright green rim + blades)
    green_lo = np.array([35, 40, 40])
    green_hi = np.array([85,255,255])

    # Dirt (brown/yellowish)
    brown_lo = np.array([10, 60, 40])
    brown_hi = np.array([30,255,255])

    mg = cv2.inRange(hsv, green_lo, green_hi)  # 0/255
    mb = cv2.inRange(hsv, brown_lo, brown_hi)

    # Reduce speckles; ignore sky/top GUI area
    h, w = mg.shape
    mg[:h//4, :] = 0
    mb[:h//4, :] = 0

    mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    return (mg > 0).astype(np.uint8), (mb > 0).astype(np.uint8)

# ---------- NEW: straight-down raycast to ground ----------
def vertical_distance_to_ground(image_bgr: np.ndarray, cx: int, cy: int):
    """
    Casts a ray straight DOWN from (cx, cy) until it hits ground (green or brown).
    Returns (dist_pixels, hit_point(x,y)). If no hit, returns (None, None).
    """
    mask_green, mask_brown = _green_brown_masks(image_bgr)
    ground = (mask_green | mask_brown).astype(np.uint8)

    h, w = ground.shape
    cx = int(np.clip(cx, 0, w-1))
    cy = int(np.clip(cy, 0, h-1))

    # Scan ONLY downward to avoid skin/sky/anything above
    hit_y = None
    for y in range(cy, h):
        if ground[y, cx] == 1:
            # Optional: prefer the boundary between green and brown if we’re on the green lip
            # advance while still on green to reach the green->brown boundary
            if mask_green[y, cx] == 1:
                y2 = y
                while y2+1 < h and mask_green[y2+1, cx] == 1:
                    y2 += 1
                hit_y = y2  # last green pixel (rim); boundary just below
            else:
                hit_y = y
            break

    if hit_y is None:
        return None, None

    return float(hit_y - cy), (cx, int(hit_y))

# ---------- Annotate (keeps all your features + new vertical height) ----------
def annotate_with_direction_line_and_ground(image_bgr: np.ndarray):
    res = angle_from_red_contour(image_bgr)
    annotated = image_bgr.copy()
    if res is None:
        cv2.putText(annotated, "Red jeep not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        return annotated, None, None

    angle, (cx, cy), cnt = res
    h, w = image_bgr.shape[:2]

    # Jeep contour
    cv2.drawContours(annotated, [cnt], -1, (0,255,0), 2)

    # Red infinite direction line (existing)
    rad = math.radians(angle)
    length = int(max(w, h) * 1.2)
    dx, dy = int(math.cos(rad)*length), int(math.sin(rad)*length)
    cv2.line(annotated, (cx-dx, cy-dy), (cx+dx, cy+dy), (0,0,255), 3)

    # Angle label
    angle_label = f"Angle: {angle:.1f} deg"
    cv2.putText(annotated, angle_label, (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(annotated, angle_label, (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # NEW: straight-down height
    dist_px, hit = vertical_distance_to_ground(image_bgr, cx, cy)
    if hit is not None:
        # Draw vertical yellow line from centroid straight down to hit
        cv2.line(annotated, (cx, cy), hit, (0, 255, 255), 2)
        cv2.circle(annotated, hit, 5, (0, 255, 255), -1)

    # Centroid marker
    cv2.circle(annotated, (cx, cy), 6, (255,255,255), -1)

    # Height label (only downward; clamped >= 0)
    height_label = f"Ground dist: {0.0 if dist_px is None else max(0.0, dist_px):.1f} px"
    cv2.putText(annotated, height_label, (cx + 10, min(h - 10, cy + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(annotated, height_label, (cx + 10, min(h - 10, cy + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return annotated, angle, dist_px

# ---------- Batch over a folder ----------
def main():
    input_dir = Path("config_screenshots/images_for_cv")
    output_dir = Path("processed_images_cv")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]
    if not files:
        print(f"No image files found in {input_dir}")
        return

    for p in sorted(files):
        print(f"Processing {p.name} ...")
        img = cv2.imread(str(p))
        if img is None:
            print("  ⚠️ Skipping (unable to read).")
            continue

        annotated, angle, dist_px = annotate_with_direction_line_and_ground(img)
        out = output_dir / f"{p.stem}_processed{p.suffix}"
        ok = cv2.imwrite(str(out), annotated)
        if ok:
            print(f"  ✅ Saved {out.name} | angle={angle} | ground_down_px={None if dist_px is None else round(dist_px,1)}")
        else:
            print(f"  ❌ Failed to save {out.name}")

    print("\nAll images processed.")

if __name__ == "__main__":
    main()
