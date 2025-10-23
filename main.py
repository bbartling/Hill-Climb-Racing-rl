import os
import csv
import time
import math
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

# Optional deps for record/preview/select modes
try:
    import keyboard
    from mss import mss
except ImportError:
    print("Warning: 'keyboard' or 'mss' not installed. Record/preview modes disabled.")
    keyboard = None
    mss = None

# Optional deps for OCR
try:
    import pytesseract
    from PIL import Image

    # --- TESSERACT INSTALLATION CHECK ---
    # You must install the Tesseract-OCR engine separately for this to work.
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
    # Mac (brew install tesseract), Linux (sudo apt-get install tesseract-ocr)
    #
    # After installing, you might need to tell pytesseract where to find it:
    # On Windows, it might be:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # --- UPDATE THIS LINE ---
    # This is the standard, non-protected path for Tesseract.
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # -------------------------------------

except ImportError:
    print("--- WARNING ---")
    print("Pytesseract or PIL (Pillow) not installed.")
    print("Install them with 'pip install pytesseract Pillow'")
    print("You also MUST install Google's Tesseract-OCR engine on your system.")
    print("See: https://github.com/tesseract-ocr/tesseract")
    print("Distance/speed OCR features will be disabled.")
    print("---------------")
    pytesseract = None
    Image = None


# Now imports the augmented function
from utils_cv import annotate_frame


# -----------------------------
# OCR Helper Function
# -----------------------------
def parse_distance_from_image(frame: np.ndarray, distance_region: dict | None):
    """
    Uses Pytesseract to OCR the distance (e.g., '322m') from a cropped region.
    Returns (int_distance, raw_ocr_text) or (None, None)
    """
    global pytesseract  # <-- Ensures we can modify the global var on error
    
    if pytesseract is None or Image is None or distance_region is None:
        return None, None

    try:
        # Crop to the specified region
        l, t, w, h = (
            distance_region["left"],
            distance_region["top"],
            distance_region["width"],
            distance_region["height"],
        )

        # Ensure crop is valid within the frame
        h_frame, w_frame = frame.shape[:2]
        if t < 0 or l < 0 or t + h > h_frame or l + w > w_frame:
            # print(f"Warning: Distance region {distance_region} is outside frame bounds {w_frame}x{h_frame}")
            return None, None

        crop = frame[t : t + h, l : l + w]

        # Pre-processing for OCR:
        # Isolate *white* pixels using a BGR color mask.
        # This is more robust than grayscale thresholding if the background is also bright.
        white_mask = cv2.inRange(crop, (200, 200, 200), (255, 255, 255))

        # Invert the mask to get black text on a white background (better for Tesseract)
        ocr_img = cv2.bitwise_not(white_mask)

        # Convert to PIL Image
        img_pil = Image.fromarray(ocr_img)

        # --psm 7: Treat the image as a single line of text.
        # Whitelist only numbers and the letter 'm'.
        config = r"--psm 7 -c tessedit_char_whitelist=0123456789m"
        raw_text = pytesseract.image_to_string(img_pil, config=config).strip()

        # Clean text to get only digits
        cleaned_text = re.sub(r"[^0-9]", "", raw_text)
        if not cleaned_text:
            return None, raw_text

        return int(cleaned_text), raw_text

    except Exception as e:
        # This can happen if Tesseract isn't installed correctly
        print(f"Error in OCR: {e}. Is Tesseract engine installed and in PATH?")
        # We can now safely assign to the global variable
        pytesseract = None  # Disable further attempts
        return None, None


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
                dist_label = f"Dist: {distance}m" if distance is not None else "Dist: None"
                raw_label = f"({raw_text})" if raw_text else "('')"

                # Draw Dist label
                cv2.putText(
                    annotated,
                    dist_label,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0), # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    dist_label,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255), # White
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
                    (0, 0, 0), # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    raw_label,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), # White
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
        fps = 30.0

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
                "speed_mps",  # New speed column
                "gas_pressed",
                "brake_pressed",
            ]
        )

    print(f"Annotating video -> {video_out} | {w}x{h} @ {fps:.2f} FPS")

    # --- Process first frame ---
    i = 0
    timestamp_s = i / fps
    annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

    distance, raw_text = parse_distance_from_image(frame, distance_region)
    speed_mps = None  # Can't calculate speed on first frame
    prev_distance = distance
    prev_timestamp_s = timestamp_s

    # Add OCR info
    if distance is not None:
        cv2.putText(
            annotated,
            f"Dist: {distance}m",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0), # Shadow
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Dist: {distance}m",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255), # White
            2,
            cv2.LINE_AA,
        )
    if raw_text:
        cv2.putText(
            annotated,
            f"({raw_text})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Shadow
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"({raw_text})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255), # White
            1,
            cv2.LINE_AA,
        )
    # Speed text (None on first frame)
    cv2.putText(
        annotated,
        "Speed: N/A",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0), # Shadow
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Speed: N/A",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255), # White
        2,
        cv2.LINE_AA,
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
                distance if distance is not None else "",
                speed_mps if speed_mps is not None else "",
                0,
                0,
            ]
        )
    i += 1

    # --- Continue with the rest ---
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = i / fps
        annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

        # --- New OCR & Speed Logic ---
        distance, raw_text = parse_distance_from_image(frame, distance_region)
        speed_mps = None

        if (
            distance is not None
            and prev_distance is not None
            and prev_timestamp_s is not None
        ):
            time_delta = timestamp_s - prev_timestamp_s
            dist_delta = distance - prev_distance
            # Plausibility check: positive time, reasonable distance change
            if time_delta > 0 and abs(dist_delta) < 100:
                speed_mps = dist_delta / time_delta

        # Add OCR info
        if distance is not None:
            cv2.putText(
                annotated,
                f"Dist: {distance}m",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0), # Shadow
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Dist: {distance}m",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255), # White
                2,
                cv2.LINE_AA,
            )
        if raw_text:
            cv2.putText(
                annotated,
                f"({raw_text})",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0), # Shadow
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"({raw_text})",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), # White
                1,
                cv2.LINE_AA,
            )
        
        # --- ADDED SPEED TEXT BLOCK ---
        speed_text = f"Speed: {speed_mps:.1f} m/s" if speed_mps is not None else "Speed: N/A"
        if speed_mps is not None:
            cv2.putText(
                annotated,
                speed_text,
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0), # Shadow
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                speed_text,
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255), # White
                2,
                cv2.LINE_AA,
            )
        # --- End New Logic ---

        writer_v.write(annotated)

        if writer_c:
            writer_c.writerow(
                [
                    f"{timestamp_s:.3f}",
                    i,
                    jeep_angle if jeep_angle is not None else "",
                    height_px if height_px is not None else "",
                    ground_slope if ground_slope is not None else "",
                    distance if distance is not None else "",
                    speed_mps if speed_mps is not None else "",
                    0,
                    0,
                ]
            )

        # Update prev values for next loop
        if distance is not None:
            prev_distance = distance
        prev_timestamp_s = timestamp_s

        if i % 60 == 0:
            print(f"  {i} frames...")
        i += 1

    cap.release()
    writer_v.release()
    if csv_file:
        csv_file.close()
    print("✅ Video saved:", video_out)


def record_mode(
    csv_path: Path,
    no_display: bool,
    game_region: dict,
    fps: float,
    video_out: Path | None,
    distance_region: dict | None,  # New arg
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
                "speed_mps",  # New columns
                "gas_pressed",
                "brake_pressed",
            ]
        )

    delay_ms = int(1000 / max(1.0, fps))
    target_delay_s = 1.0 / max(1.0, fps)
    writer_v = None

    # For speed calculation
    prev_distance = None
    prev_timestamp = None

    with mss() as sct:
        try:
            while True:
                loop_start_time = time.time()
                if keyboard.is_pressed("q"):
                    print("\n⏹️ Recording stopped.")
                    break

                sct_img = sct.grab(game_region)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                current_timestamp = time.time()

                annotated, jeep_angle, height_px, ground_slope = annotate_frame(frame)

                # --- New OCR & Speed Logic ---
                distance, raw_text = parse_distance_from_image(frame, distance_region)
                speed_mps = None

                if (
                    distance is not None
                    and prev_distance is not None
                    and prev_timestamp is not None
                ):
                    time_delta = current_timestamp - prev_timestamp
                    dist_delta = distance - prev_distance
                    if time_delta > 0 and abs(dist_delta) < 100:
                        speed_mps = dist_delta / time_delta

                gas = 1 if keyboard.is_pressed("right") else 0
                brake = 1 if keyboard.is_pressed("left") else 0

                writer.writerow(
                    [
                        current_timestamp,
                        jeep_angle if jeep_angle is not None else "",
                        height_px if height_px is not None else "",
                        ground_slope if ground_slope is not None else "",
                        distance if distance is not None else "",
                        speed_mps if speed_mps is not None else "",
                        gas,
                        brake,
                    ]
                )

                # Add overlays *before* video write / display
                if distance is not None:
                    cv2.putText(
                        annotated,
                        f"Dist: {distance}m",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0), # Shadow
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        f"Dist: {distance}m",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255), # White
                        2,
                        cv2.LINE_AA,
                    )
                if raw_text:
                    cv2.putText(
                        annotated,
                        f"({raw_text})",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0), # Shadow
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        f"({raw_text})",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255), # White
                        1,
                        cv2.LINE_AA,
                    )
                
                speed_text = f"Speed: {speed_mps:.1f} m/s" if speed_mps is not None else "Speed: N/A"
                if speed_mps is not None:
                    cv2.putText(
                        annotated,
                        speed_text,
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0), # Shadow
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        speed_text,
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255), # White
                        2,
                        cv2.LINE_AA,
                    )
                # --- End New Logic ---

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

                if not no_display:
                    # We already modified 'annotated', so just add Action text
                    action_text = "GAS" if gas else ("BRAKE" if brake else "COAST")
                    cv2.putText(
                        annotated,  # Use 'annotated' directly
                        f"Action: {action_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    
                    # --- REMOVED REDUNDANT TEXT DRAWING ---

                    cv2.imshow("HCR Recorder (press 'q' to stop)", annotated)
                    # Use a calculated delay to try and match target FPS
                    loop_end_time = time.time()
                    elapsed_s = loop_end_time - loop_start_time
                    wait_s = target_delay_s - elapsed_s
                    wait_ms = max(1, int(wait_s * 1000))

                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        break
                else:
                    # Headless, sleep to respect FPS
                    loop_end_time = time.time()
                    elapsed_s = loop_end_time - loop_start_time
                    wait_s = target_delay_s - elapsed_s
                    if wait_s > 0:
                        time.sleep(wait_s)

                # Update prev values for next loop
                if distance is not None:
                    prev_distance = distance
                prev_timestamp = current_timestamp

        finally:
            csv_file.close()
            if writer_v:
                writer_v.release()
                print(f"🎥 Video saved to {video_out}")
            if not no_display:
                cv2.destroyAllWindows()
            print(f"Data saved to {csv_path}")


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
                    (l, t + h + 20), # Position text below the red box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0), # Shadow
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    ocr_label,
                    (l, t + h + 20), # Position text below the red box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255), # White
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
    cv2.setWindowProperty("Select Region", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    r = cv2.selectROI("Select Region", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")
    
    x, y, w, h = map(int, r)
    if w == 0 or h == 0: # User pressed ESC
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
        process_images(
            args.images_in, args.images_out, args.csv, distance_region_dict
        )
        return
    if args.video_in:
        process_video(
            args.video_in, args.video_out, args.csv, distance_region_dict
        )
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

