
import os
import csv
import time
import math
import argparse
from pathlib import Path

import cv2
import numpy as np

# Optional deps for record mode
try:
    import keyboard
    from mss import mss
except Exception:
    keyboard = None
    mss = None

from utils_cv import annotate_frame  # local util file

def process_images(input_dir: Path, output_dir: Path, csv_path: Path | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"No image files found in {input_dir}")
        return

    writer = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'source', 'angle', 'height_px', 'gas_pressed', 'brake_pressed'])

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
            writer.writerow([time.time(), p.name, angle if angle is not None else '', height_px if height_px is not None else '', 0, 0])

    if csv_file:
        csv_file.close()

def process_video(video_in: Path, video_out: Path | None, csv_path: Path | None):
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_out is None:
        video_out = video_in.with_name(f"{video_in.stem}_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_v = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer_v.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {video_out}")

    writer_c = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer_c = csv.writer(csv_file)
        writer_c.writerow(['timestamp_s', 'frame_idx', 'angle', 'height_px', 'gas_pressed', 'brake_pressed'])

    i = 0
    print(f"Annotating video -> {video_out} | {width}x{height} @ {fps:.2f} FPS")
    while True:
        ok, frame = cap.read();
        if not ok:
            break
        annotated, angle, height_px = annotate_frame(frame)
        writer_v.write(annotated)

        if writer_c:
            t = i / float(fps)
            writer_c.writerow([f"{t:.3f}", i, angle if angle is not None else '', height_px if height_px is not None else '', 0, 0])

        i += 1
        if i % 60 == 0:
            print(f"  {i} frames...")

    cap.release()
    writer_v.release()
    if csv_file:
        csv_file.close()
    print("✅ Video saved:", video_out)

def record_mode(csv_path: Path, no_display: bool, game_region: dict, fps: float):
    if keyboard is None or mss is None:
        raise RuntimeError("'record' mode requires 'keyboard' and 'mss' packages installed.")

    print("--- Manual Gameplay Recorder ---")
    print("Bring the game window into focus. Press 's' to start, 'q' to quit.")
    print(f"Region: {game_region}")
    keyboard.wait('s')
    print("▶️ Recording started! Press 'q' to quit.")

    file_exists = csv_path.exists()
    csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(['timestamp', 'angle', 'height_px', 'gas_pressed', 'brake_pressed'])

    delay = 1.0 / max(1.0, fps)

    with mss() as sct:
        try:
            while True:
                if keyboard.is_pressed('q'):
                    print("\n⏹️ Recording stopped.")
                    break

                sct_img = sct.grab(game_region)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                annotated, angle, height_px = annotate_frame(frame)

                gas = 1 if keyboard.is_pressed('right') else 0
                brake = 1 if keyboard.is_pressed('left') else 0

                writer.writerow([time.time(), angle if angle is not None else '', height_px if height_px is not None else '', gas, brake])

                if not no_display:
                    # Light overlay for live preview
                    disp = annotated.copy()
                    action_text = 'GAS' if gas else ('BRAKE' if brake else 'COAST')
                    cv2.putText(disp, f"Action: {action_text}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    cv2.imshow("HCR Recorder (press 'q' to stop)", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                time.sleep(delay)
        finally:
            csv_file.close()
            if not no_display:
                cv2.destroyAllWindows()
            print(f"Data saved to {csv_path}")

def parse_args():
    p = argparse.ArgumentParser(description='Hill Climb Racing CV tools: images | video | record')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--images-in', type=Path, help='Directory of images to process')
    mode.add_argument('--video-in', type=Path, help='Video file to process')
    mode.add_argument('--record', action='store_true', help='Record manual gameplay to CSV')

    # Common / optional outputs
    p.add_argument('--images-out', type=Path, default=Path('processed_images_cv'), help='Output dir for annotated images')
    p.add_argument('--video-out', type=Path, help='Output path for annotated MP4')
    p.add_argument('--csv', type=Path, help='Optional CSV path to write angle/height (+ gas/brake when recording)')

    # Record options
    p.add_argument('--no-display', action='store_true', help='Headless (no preview window) in record mode')
    p.add_argument('--fps', type=float, default=20.0, help='Sampling FPS in record mode (default: 20.0)')
    p.add_argument('--region', type=int, nargs=4, metavar=('LEFT','TOP','WIDTH','HEIGHT'),
                   help='Screen region for HCR window in record mode, e.g. --region 68 35 1235 687')

    return p.parse_args()

def main():
    args = parse_args()
    if args.images_in:
        process_images(args.images_in, args.images_out, args.csv)
    elif args.video_in:
        process_video(args.video_in, args.video_out, args.csv)
    elif args.record:
        if args.csv is None:
            raise SystemExit("Please provide --csv for record mode.")
        # region mapping for mss
        if args.region and len(args.region) == 4:
            left, top, width, height = args.region
            region = {'left': left, 'top': top, 'width': width, 'height': height}
        else:
            # sensible default (edit to your monitor layout)
            region = {'left': 68, 'top': 35, 'width': 1235, 'height': 687}
        record_mode(args.csv, args.no_display, region, args.fps)

if __name__ == '__main__':
    main()
