import os
import csv
import time
import math
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Tuple, Optional


import keyboard
from mss import mss



pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# Now imports the augmented function
from utils_cv import annotate_frame


def parse_distance_from_image(frame_bgr: np.ndarray,
                              distance_region: dict
                              ) -> Tuple[Optional[int], str]:
    """
    Try hard to OCR the distance HUD.
    Returns:
        distance_int (int | None), raw_text (str)
    Strategy:
      - crop HUD box
      - preprocess for high-contrast digits
      - upsample
      - run tesseract in 2 passes (strict + fallback)
    """

    # Safety: bad region
    if distance_region is None:
        return None, ""

    l = distance_region["left"]
    t = distance_region["top"]
    w = distance_region["width"]
    h = distance_region["height"]
    sub = frame_bgr[t:t + h, l:l + w]

    if sub.size == 0:
        return None, ""

    # ---------- helper: run tesseract and parse digits ----------
    def ocr_and_parse(bin_img: np.ndarray, psm: int) -> Tuple[Optional[int], str]:
        # Upscale (helps a TON for tiny HUD fonts)
        scale = 3
        big = cv2.resize(
            bin_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        pil_img = Image.fromarray(big)

        config = (
            f"--psm {psm} "
            "-c tessedit_char_whitelist=0123456789m "
            "-c classify_bln_numeric_mode=1"
        )
        raw = pytesseract.image_to_string(pil_img, config=config).strip()

        # grab only digits because we only care about distance number
        just_digits = re.sub(r"[^0-9]", "", raw)
        if just_digits == "":
            dist_val = None
        else:
            try:
                dist_val = int(just_digits)
            except ValueError:
                dist_val = None

        return dist_val, raw

    # ---------- STEP 1: grayscale + adaptive threshold ----------
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

    # adaptive threshold handles lighting flicker / glow
    # we invert so we end up "black text on white" after morphology later
    # Note: blockSize MUST be odd and ~ digit height-ish. 15-31 works well.
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # so digits become white blobs
        21,
        10,
    )

    # ---------- STEP 2: biggest contour crop (so we don't feed sky / hills) ----------
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)

        pad = 4
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + bw + pad, thr.shape[1])
        y1 = min(y + bh + pad, thr.shape[0])
        roi = thr[y0:y1, x0:x1]
    else:
        roi = thr

    # ---------- STEP 3: thicken strokes & clean speckles ----------
    # We have white digits on black (because of THRESH_BINARY_INV above).
    # We'll close tiny gaps so "1234" looks solid.
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    # light blur can smooth jaggy edges so tess stops hallucinating extra tails
    smooth = cv2.GaussianBlur(closed, (3, 3), 0)

    # Invert for tesseract: black digits on white background
    prep_main = cv2.bitwise_not(smooth)

    # ---------- PASS A (strict): psm 7 (single line) ----------
    dist_a, raw_a = ocr_and_parse(prep_main, psm=7)

    # ---------- PASS B (fallback): more forgiving  --psm 13 ----------
    # Sometimes adaptiveThreshold nukes shadows too hard, so let's also try
    # a slightly looser binarization (global Otsu) and a different psm.
    if dist_a is None:
        # global Otsu on grayscale crop of JUST the ROI box for fallback
        roi_gray = gray[y0:y1, x0:x1] if contours else gray
        # use Otsu, invert so digits become white blobs
        _, thr_otsu = cv2.threshold(
            roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # re-close to thicken
        closed2 = cv2.morphologyEx(thr_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        smooth2 = cv2.GaussianBlur(closed2, (3, 3), 0)
        prep_fallback = cv2.bitwise_not(smooth2)

        dist_b, raw_b = ocr_and_parse(prep_fallback, psm=13)

        # choose best
        if dist_b is not None:
            return dist_b, raw_b
        else:
            return None, raw_b  # raw_b might still be useful debug text

    # PASS A worked
    return dist_a, raw_a


# -----------------------------
# Core processing functions
# -----------------------------
def process_images(
    input_dir: Path,
    output_dir: Path,
    csv_path: Path | None,
    distance_region: dict | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"No image files found in {input_dir}")
        return

    writer = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp",
                "source",
                "jeep_angle",
                "height_px",
                "ground_slope",
                "distance_m",
                "speed_mps",  # Speed column
                "gas_pressed",
                "brake_pressed",
            ]
        )

    for p in sorted(files):
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {p.name}")
            continue

        annotated, jeep_angle, height_px, ground_slope = annotate_frame(img)

        # --- New OCR & Speed Logic ---
        distance, raw_text = parse_distance_from_image(img, distance_region)
        speed_mps = None  # Cannot calculate speed from a single image

        # --- Draw Distance Region Box and Text (ALWAYS) ---
        if distance_region:
            try:
                l, t, w, h = (
                    distance_region["left"],
                    distance_region["top"],
                    distance_region["width"],
                    distance_region["height"],
                )
                # Draw the red box
                cv2.rectangle(annotated, (l, t), (l + w, t + h), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    "Distance Region",
                    (l, max(0, t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                # Format the text to show, even if OCR failed
                dist_label = (
                    f"Dist: {distance}m" if distance is not None else "Dist: None"
                )
                raw_label = f"({raw_text})" if raw_text else "('')"

                # Draw Dist label
                cv2.putText(
                    annotated,
                    dist_label,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),  # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    dist_label,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White
                    2,
                    cv2.LINE_AA,
                )

                # Draw Raw label
                cv2.putText(
                    annotated,
                    raw_label,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    raw_label,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White
                    1,
                    cv2.LINE_AA,
                )
            except Exception as e:
                print(f"Could not draw distance region on image: {e}")
        # --- End New Logic ---

        out = output_dir / f"{p.stem}_processed{p.suffix}"
        cv2.imwrite(str(out), annotated)
        print(
            f"✅ Saved {out.name} | angle={jeep_angle} | height_px={height_px} | slope={ground_slope} | dist={distance}"
        )

        if writer:
            writer.writerow(
                [
                    time.time(),
                    p.name,
                    jeep_angle if jeep_angle is not None else "",
                    height_px if height_px is not None else "",
                    ground_slope if ground_slope is not None else "",
                    distance if distance is not None else "",
                    speed_mps if speed_mps is not None else "",
                    0,
                    0,
                ]
            )

    if csv_file:
        csv_file.close()


def record_mode(
    csv_path: Path,
    no_display: bool,
    game_region: dict,
    fps: float,
    video_out: Path | None,
    distance_region: dict | None,
):
    if keyboard is None or mss is None:
        raise RuntimeError(
            "'record' mode requires 'keyboard' and 'mss' packages installed."
        )

    print("--- Manual Gameplay Recorder ---")
    print("Bring the game window into focus. Press 's' to start, 'q' to quit.")
    print(f"Region: {game_region}")
    if distance_region:
        print(f"Distance Region: {distance_region}")
    keyboard.wait("s")
    print("▶️ Recording started! Press 'q' to quit.")

    # CSV setup / append
    file_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(
            [
                "timestamp",
                "jeep_angle",
                "height_px",
                "ground_slope",
                "distance_m",
                "speed_mps",
                "gas_pressed",
                "brake_pressed",
            ]
        )

    delay_ms = int(1000 / max(1.0, fps))
    target_delay_s = 1.0 / max(1.0, fps)

    writer_v = None

    # rolling state for speed calc
    prev_distance = None
    prev_timestamp = None

    with mss() as sct:
        try:
            while True:
                loop_start_time = time.time()

                # quit hotkey
                if keyboard.is_pressed("q"):
                    print("\n⏹️ Recording stopped.")
                    break

                # grab frame from screen
                sct_img = sct.grab(game_region)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                current_timestamp = time.time()

                # run vision annotator (angle, height, slope lines, etc)
                annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

                # OCR HUD read
                distance_val, raw_text = parse_distance_from_image(frame, distance_region)

                # distance sanity clamp (kill insane OCR)
                if distance_val is not None and distance_val > 20000:
                    distance_val = None

                # pedal state (ground truth labels)
                gas_pressed = 1 if keyboard.is_pressed("right") else 0
                brake_pressed = 1 if keyboard.is_pressed("left") else 0

                # estimate speed from distance deltas
                speed_mps = None
                if (
                    distance_val is not None
                    and prev_distance is not None
                    and prev_timestamp is not None
                ):
                    time_delta = current_timestamp - prev_timestamp
                    dist_delta = distance_val - prev_distance
                    if time_delta > 0 and abs(dist_delta) < 100:
                        speed_mps = dist_delta / time_delta

                # write CSV row for training data
                writer.writerow(
                    [
                        current_timestamp,
                        jeep_angle if jeep_angle is not None else "",
                        height_px if height_px is not None else "",
                        ground_slope if ground_slope is not None else "",
                        distance_val if distance_val is not None else "",
                        speed_mps if speed_mps is not None else "",
                        gas_pressed,
                        brake_pressed,
                    ]
                )

                # HUD overlay
                annotated = draw_hud(
                    annotated=annotated,
                    jeep_angle=jeep_angle,
                    height_px=height_px,
                    ground_slope=ground_slope,
                    distance=distance_val,
                    raw_text=raw_text,
                    speed_mps=speed_mps,
                    gas=gas_pressed,
                    brake=brake_pressed,
                    distance_region=distance_region,
                )

                # Init video writer on first frame if needed
                if video_out:
                    if writer_v is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer_v = cv2.VideoWriter(
                            str(video_out),
                            fourcc,
                            fps,
                            (annotated.shape[1], annotated.shape[0]),
                        )
                        if not writer_v.isOpened():
                            raise RuntimeError(
                                f"Could not open VideoWriter: {video_out}"
                            )
                    writer_v.write(annotated)

                # live preview window unless --no-display
                if not no_display:
                    cv2.imshow("HCR Recorder (press 'q' to stop)", annotated)

                    # try to throttle loop to requested FPS
                    loop_end_time = time.time()
                    elapsed_s = loop_end_time - loop_start_time
                    wait_s = target_delay_s - elapsed_s
                    wait_ms = max(1, int(wait_s * 1000))

                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        break
                else:
                    # headless, just sleep to respect FPS
                    loop_end_time = time.time()
                    elapsed_s = loop_end_time - loop_start_time
                    wait_s = target_delay_s - elapsed_s
                    if wait_s > 0:
                        time.sleep(wait_s)

                # update for next speed calc
                if distance_val is not None:
                    prev_distance = distance_val
                    prev_timestamp = current_timestamp
                # if OCR failed / clamped, don't advance prev_*

        finally:
            csv_file.close()
            if writer_v:
                writer_v.release()
                print(f"🎥 Video saved to {video_out}")
            if not no_display:
                cv2.destroyAllWindows()
            print(f"Data saved to {csv_path}")



def process_video(
    video_in: Path,
    video_out: Path | None,
    csv_path: Path | None,
    distance_region: dict | None,
):
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # fallback

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame; aborting.")

    h, w = frame.shape[:2]

    if video_out is None:
        video_out = video_in.with_name(f"{video_in.stem}_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_v = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))
    if not writer_v.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {video_out}")

    # CSV setup
    writer_c = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer_c = csv.writer(csv_file)
        writer_c.writerow(
            [
                "timestamp_s",
                "frame_idx",
                "jeep_angle",
                "height_px",
                "ground_slope",
                "distance_m",
                "speed_mps",
                "gas_pressed",
                "brake_pressed",
            ]
        )

    print(f"Annotating video -> {video_out} | {w}x{h} @ {fps:.2f} FPS")

    # ---- first frame ----
    i = 0
    timestamp_s = i / fps

    annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

    distance_val, raw_text = parse_distance_from_image(frame, distance_region)

    # clamp insane OCR
    if distance_val is not None and distance_val > 20000:
        distance_val = None

    speed_mps = None
    prev_distance = distance_val
    prev_timestamp_s = timestamp_s

    annotated = draw_hud(
        annotated=annotated,
        jeep_angle=jeep_angle,
        height_px=height_px,
        ground_slope=ground_slope,
        distance=distance_val,
        raw_text=raw_text,
        speed_mps=speed_mps,
        gas=0,
        brake=0,
        distance_region=distance_region,
    )

    writer_v.write(annotated)

    if writer_c:
        writer_c.writerow(
            [
                f"{timestamp_s:.3f}",
                i,
                jeep_angle if jeep_angle is not None else "",
                height_px if height_px is not None else "",
                ground_slope if ground_slope is not None else "",
                distance_val if distance_val is not None else "",
                speed_mps if speed_mps is not None else "",
                0,
                0,
            ]
        )

    i += 1

    # ---- rest of frames ----
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = i / fps

        annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

        distance_val, raw_text = parse_distance_from_image(frame, distance_region)

        # clamp insane OCR
        if distance_val is not None and distance_val > 20000:
            distance_val = None

        speed_mps = None
        if (
            distance_val is not None
            and prev_distance is not None
            and prev_timestamp_s is not None
        ):
            dt = timestamp_s - prev_timestamp_s
            dist_delta = distance_val - prev_distance
            if dt > 0 and abs(dist_delta) < 100:
                speed_mps = dist_delta / dt

        annotated = draw_hud(
            annotated=annotated,
            jeep_angle=jeep_angle,
            height_px=height_px,
            ground_slope=ground_slope,
            distance=distance_val,
            raw_text=raw_text,
            speed_mps=speed_mps,
            gas=0,
            brake=0,
            distance_region=distance_region,
        )

        writer_v.write(annotated)

        if writer_c:
            writer_c.writerow(
                [
                    f"{timestamp_s:.3f}",
                    i,
                    jeep_angle if jeep_angle is not None else "",
                    height_px if height_px is not None else "",
                    ground_slope if ground_slope is not None else "",
                    distance_val if distance_val is not None else "",
                    speed_mps if speed_mps is not None else "",
                    0,
                    0,
                ]
            )

        # only advance prev_* if OCR didn't glitch
        if distance_val is not None:
            prev_distance = distance_val
            prev_timestamp_s = timestamp_s

        if i % 60 == 0:
            print(f"  {i} frames...")

        i += 1

    cap.release()
    writer_v.release()
    if csv_file:
        csv_file.close()
    print("✅ Video saved:", video_out)




# -----------------------------
# Region utilities
# -----------------------------
def preview_region(region: dict, distance_region: dict | None):
    """Grab one frame of the specified region and display it once."""
    if mss is None:
        raise RuntimeError("preview requires 'mss' installed.")
    with mss() as sct:
        sct_img = sct.grab(region)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # Draw main region border (Yellow)
        cv2.rectangle(
            frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Region {region['left']},{region['top']} {region['width']}x{region['height']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --- New: Draw distance region (Red) ---
        if distance_region:
            try:
                # These coords are relative to the main region, which is our frame
                l, t, w, h = (
                    distance_region["left"],
                    distance_region["top"],
                    distance_region["width"],
                    distance_region["height"],
                )
                cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Distance Region",
                    (l, max(0, t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                # --- REMOVED BAD SPEED TEXT LINE ---

                # --- ADDING OCR TEST ---
                # Run OCR on this specific region from the captured frame
                # The 'frame' is already the cropped main region, so distance_region coords are correct
                distance, raw_text = parse_distance_from_image(frame, distance_region)

                print("\n--- OCR TEST RESULTS ---")
                print(f"Raw Text: '{raw_text}'")
                print(f"Parsed Distance: {distance}")
                print("------------------------")
                print("Press any key in the preview window to close...")

                # Draw the OCR results onto the preview window
                ocr_label = f"OCR: {distance}m ('{raw_text}')"
                cv2.putText(
                    frame,
                    ocr_label,
                    (l, t + h + 20),  # Position text below the red box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),  # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    ocr_label,
                    (l, t + h + 20),  # Position text below the red box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White
                    2,
                    cv2.LINE_AA,
                )
                # --- END OCR TEST ---

            except Exception as e:
                print(f"Could not draw/OCR distance region: {e}")
        # --- End New ---

        cv2.imshow("Region Snapshot", frame)
        print("Showing snapshot. Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def select_region_interactive(save_to: Path | None = None):
    """Take a full-screen screenshot and use OpenCV's selectROI to interactively pick a region.
    Returns dict {'left','top','width','height'}.
    """
    if mss is None:
        raise RuntimeError("select-region requires 'mss' installed.")
    with mss() as sct:
        # monitor 1 == primary full screen
        mon = sct.monitors[1]
        sct_img = sct.grab(mon)
        full = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    clone = full.copy()
    cv2.putText(
        clone,
        "Drag to select region, ENTER to confirm, ESC to cancel",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        clone,
        "Drag to select region, ENTER to confirm, ESC to cancel",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # Use main window
    cv2.namedWindow("Select Region", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Select Region", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    r = cv2.selectROI("Select Region", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")

    x, y, w, h = map(int, r)
    if w == 0 or h == 0:  # User pressed ESC
        print("Region selection cancelled.")
        return None

    # These are global screen.
    region = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
    print(
        f"Selected region: {region['left']} {region['top']} {region['width']} {region['height']}"
    )

    if save_to:
        save_to.write_text(
            f"{region['left']} {region['top']} {region['width']} {region['height']}",
            encoding="utf-8",
        )
        print(f"Saved to {save_to.resolve()}")
    return region


def draw_hud(
    annotated,
    jeep_angle,
    height_px,
    ground_slope,
    distance,
    raw_text,
    speed_mps,
    gas,
    brake,
    distance_region,
):
    """
    Draw ALL overlay text/boxes in a consistent way so both preview,
    record, and post-process videos look identical.
    """

    # 1. Draw the distance-region capture box in red so it's visible in output video
    if distance_region is not None:
        l = distance_region["left"]
        t = distance_region["top"]
        w = distance_region["width"]
        h = distance_region["height"]
        cv2.rectangle(annotated, (l, t), (l + w, t + h), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "Distance Region",
            (l, max(0, t - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 2. Decide action text
    if gas:
        action_text = "GAS"
    elif brake:
        action_text = "BRAKE"
    else:
        action_text = "COAST"

    # 3. Overlay block (top-left corner HUD)
    # Action line
    cv2.putText(
        annotated,
        f"Action: {action_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Distance line (shadowed white text for readability)
    dist_label = f"Dist: {distance}m" if distance is not None else "Dist: None"
    cv2.putText(
        annotated,
        dist_label,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        dist_label,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Raw OCR text (for debugging OCR quality)
    raw_label = f"({raw_text})" if raw_text else "('')"
    cv2.putText(
        annotated,
        raw_label,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        raw_label,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Speed line
    speed_label = (
        f"Speed: {speed_mps:.1f} m/s" if speed_mps is not None else "Speed: N/A"
    )
    cv2.putText(
        annotated,
        speed_label,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        speed_label,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Note: jeep_angle, height_px, and ground_slope are already rendered
    # by annotate_frame() itself in your code, so we don't redraw them here.

    return annotated


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Hill Climb Racing CV tools: images | video | record | region tools"
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images-in", type=Path, help="Directory of images to process")
    mode.add_argument("--video-in", type=Path, help="Video file to process")
    mode.add_argument(
        "--record", action="store_true", help="Record manual gameplay to CSV"
    )
    mode.add_argument(
        "--preview-region",
        action="store_true",
        help="Live preview of a region (use with --region)",
    )
    mode.add_argument(
        "--select-region",
        action="store_true",
        help="Interactively select a region and print/save it",
    )

    # Common / optional outputs
    p.add_argument(
        "--images-out",
        type=Path,
        default=Path("processed_images_cv"),
        help="Output dir for annotated images",
    )
    p.add_argument("--video-out", type=Path, help="Output path for annotated MP4")
    p.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV path to write angle/height (+ gas/brake when recording)",
    )

    # Record options
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Headless (no preview window) in record mode",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Sampling FPS in record mode (default: 20.0)",
    )
    p.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen region for HCR window in record/preview mode, e.g. --region 68 35 1235 687",
    )
    p.add_argument(
        "--distance-region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Region for distance OCR, relative to the main game region (e.g., 250 20 100 50)",
    )
    p.add_argument(
        "--record-video-out",
        type=Path,
        help="Optional output MP4 for annotated gameplay video (record mode)",
    )

    # Region tools options
    p.add_argument(
        "--save-region-to",
        type=Path,
        help="When using --select-region, also save the chosen region to this text file",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # --- New: Convert distance-region to a dict ---
    distance_region_dict = None
    if args.distance_region and len(args.distance_region) == 4:
        l, t, w, h = args.distance_region
        # These coordinates are relative to the main game region
        distance_region_dict = {"left": l, "top": t, "width": w, "height": h}
    # --- End New ---

    if args.images_in:
        process_images(args.images_in, args.images_out, args.csv, distance_region_dict)
        return
    if args.video_in:
        process_video(args.video_in, args.video_out, args.csv, distance_region_dict)
        return

    # --- Region tools ---
    if args.select_region:
        select_region_interactive(args.save_region_to)
        return

    # The following modes require a main region
    main_region_dict = None
    if args.region and len(args.region) == 4:
        left, top, width, height = args.region
        main_region_dict = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }
    else:
        # User might be using --preview-region without --region, so exit
        if args.preview_region or args.record:
            raise SystemExit(
                "Please provide --region LEFT TOP WIDTH HEIGHT for --preview-region or --record modes."
            )

    if args.preview_region:
        # Pass both regions to the preview function
        preview_region(main_region_dict, distance_region_dict)
        return

    if args.record:
        if args.csv is None:
            raise SystemExit("Please provide --csv for record mode.")
        record_mode(
            args.csv,
            args.no_display,
            main_region_dict,
            args.fps,
            args.record_video_out,
            distance_region_dict,  # Pass new region
        )
        return


if __name__ == "__main__":
    main()
