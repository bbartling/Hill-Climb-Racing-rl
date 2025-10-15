import os
import csv
import time
import math
import argparse
from pathlib import Path

import cv2
import numpy as np

# Optional deps for record/preview/select modes
try:
    import keyboard
    from mss import mss
except Exception:
    keyboard = None
    mss = None

from utils_cv import annotate_frame  # local util file


# -----------------------------
# Core processing functions
# -----------------------------
def process_images(input_dir: Path, output_dir: Path, csv_path: Path | None):
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
                "angle",
                "height_px",
                "gas_pressed",
                "brake_pressed",
            ]
        )

    for p in sorted(files):
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {p.name}")
            continue
        annotated, angle, height_px = annotate_frame(img)
        out = output_dir / f"{p.stem}_processed{p.suffix}"
        cv2.imwrite(str(out), annotated)
        print(f"✅ Saved {out.name} | angle={angle} | height_px={height_px}")

        if writer:
            writer.writerow(
                [
                    time.time(),
                    p.name,
                    angle if angle is not None else "",
                    height_px if height_px is not None else "",
                    0,
                    0,
                ]
            )

    if csv_file:
        csv_file.close()


def process_video(video_in: Path, video_out: Path | None, csv_path: Path | None):
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    # FPS can be 0 or bogus; default to 30 if needed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    # Read first frame to get the TRUE size
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame; aborting.")

    h, w = frame.shape[:2]

    if video_out is None:
        video_out = video_in.with_name(f"{video_in.stem}_processed.mp4")

    # Try mp4v first; if it ever fails on your system, swap to 'avc1' or 'XVID' (.avi)
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
                "angle",
                "height_px",
                "gas_pressed",
                "brake_pressed",
            ]
        )

    print(f"Annotating video -> {video_out} | {w}x{h} @ {fps:.2f} FPS")

    # Process the already-read first frame
    i = 0
    annotated, angle, height_px = annotate_frame(frame)
    writer_v.write(annotated)
    if writer_c:
        writer_c.writerow(
            [
                f"{i / fps:.3f}",
                i,
                angle if angle is not None else "",
                height_px if height_px is not None else "",
                0,
                0,
            ]
        )
    i += 1

    # Continue with the rest
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, angle, height_px = annotate_frame(frame)
        writer_v.write(annotated)
        if writer_c:
            writer_c.writerow(
                [
                    f"{i / fps:.3f}",
                    i,
                    angle if angle is not None else "",
                    height_px if height_px is not None else "",
                    0,
                    0,
                ]
            )
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
):
    if keyboard is None or mss is None:
        raise RuntimeError(
            "'record' mode requires 'keyboard' and 'mss' packages installed."
        )

    print("--- Manual Gameplay Recorder ---")
    print("Bring the game window into focus. Press 's' to start, 'q' to quit.")
    print(f"Region: {game_region}")
    keyboard.wait("s")
    print("▶️ Recording started! Press 'q' to quit.")

    file_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(
            ["timestamp", "angle", "height_px", "gas_pressed", "brake_pressed"]
        )

    delay_ms = int(1000 / max(1.0, fps))
    writer_v = None

    with mss() as sct:
        try:
            while True:
                if keyboard.is_pressed("q"):
                    print("\n⏹️ Recording stopped.")
                    break

                sct_img = sct.grab(game_region)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                annotated, angle, height_px = annotate_frame(frame)

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

                gas = 1 if keyboard.is_pressed("right") else 0
                brake = 1 if keyboard.is_pressed("left") else 0

                writer.writerow(
                    [
                        time.time(),
                        angle if angle is not None else "",
                        height_px if height_px is not None else "",
                        gas,
                        brake,
                    ]
                )

                if not no_display:
                    disp = annotated.copy()
                    action_text = "GAS" if gas else ("BRAKE" if brake else "COAST")
                    cv2.putText(
                        disp,
                        f"Action: {action_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("HCR Recorder (press 'q' to stop)", disp)
                    # Use the calculated millisecond delay here
                    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                        break
                else:  # If running headless, we still need a delay
                    time.sleep(1.0 / fps)

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
def preview_region(region: dict):
    """Grab one frame of the specified region and display it once."""
    if mss is None:
        raise RuntimeError("preview requires 'mss' installed.")
    with mss() as sct:
        sct_img = sct.grab(region)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
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
        cv2.imshow("Region Snapshot", frame)
        print("Press any key to close...")
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
    r = cv2.selectROI("Select Region", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")
    x, y, w, h = map(int, r)
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
    # Region-tools are standalone helpers
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
    if args.images_in:
        process_images(args.images_in, args.images_out, args.csv)
        return
    if args.video_in:
        process_video(args.video_in, args.video_out, args.csv)
        return
    if args.preview_region:
        if not args.region or len(args.region) != 4:
            raise SystemExit(
                "Please provide --region LEFT TOP WIDTH HEIGHT for --preview-region."
            )
        left, top, width, height = args.region
        region = {"left": left, "top": top, "width": width, "height": height}
        preview_region(region)
        return
    if args.select_region:
        select_region_interactive(args.save_region_to)
        return
    if args.record:
        if args.csv is None:
            raise SystemExit("Please provide --csv for record mode.")
        # region mapping for mss
        if args.region and len(args.region) == 4:
            left, top, width, height = args.region
            region = {"left": left, "top": top, "width": width, "height": height}
        else:
            # sensible default (edit to your monitor layout)
            region = {"left": 68, "top": 35, "width": 1235, "height": 687}
        record_mode(args.csv, args.no_display, region, args.fps, args.record_video_out)
        return


if __name__ == "__main__":
    main()
