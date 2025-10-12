import cv2
import numpy as np
import math
import time
import os
import csv
import keyboard
from mss import mss
import argparse  # <-- 1. IMPORT a new library

# ==================================
# --- CONFIGURATION & SETUP ---
# ==================================
GAME_REGION = {"left": 68, "top": 35, "width": 1235, "height": 687}
GAS_KEY = 'right'
BRAKE_KEY = 'left'
OUTPUT_CSV = "manual_play_data.csv"

# ===============================================================
# --- ANGLE DETECTION LOGIC (borrowed from your jeep_angle.py) ---
# ===============================================================
def get_jeep_angle(image_bgr: np.ndarray):
    """Detects the red jeep and returns its angle, center, and contour."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 90, 60]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 90, 60]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200: return None
    (cx, cy), (w, h), theta = cv2.minAreaRect(cnt)
    angle = theta + 90 if w < h else theta
    return (float(angle), (int(cx), int(cy)))

# ==================================
# --- MAIN RECORDING LOOP ---
# ==================================
def main():
    """Main function to capture, annotate, and record gameplay."""
    
    # --- 2. ADD ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Record manual gameplay for Hill Climb Racing.")
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Run in headless mode without the preview window."
    )
    args = parser.parse_args()

    print("--- Manual Gameplay Recorder ---")
    if args.no_display:
        print("--- Running in HEADLESS mode (no display) ---")
    
    print(f"  Gas Key: '{GAS_KEY}' | Brake Key: '{BRAKE_KEY}'")
    print("\nBring the game window into focus.")
    print("Press 's' to start recording...")
    keyboard.wait('s')
    print("▶️ Recording started! Press 'q' to quit.")

    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["timestamp", "angle", "gas_pressed", "brake_pressed"])

    with mss() as sct:
        try:
            while True:
                if keyboard.is_pressed('q'):
                    print("\n⏹️ Recording stopped.")
                    break

                sct_img = sct.grab(GAME_REGION)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                
                current_angle = 0.0
                angle_data = get_jeep_angle(frame)
                if angle_data:
                    current_angle, _ = angle_data
                
                gas = 1 if keyboard.is_pressed(GAS_KEY) else 0
                brake = 1 if keyboard.is_pressed(BRAKE_KEY) else 0
                
                timestamp = time.time()
                writer.writerow([timestamp, current_angle, gas, brake])

                # --- 3. MAKE THE DISPLAY CONDITIONAL ---
                if not args.no_display:
                    annotated_frame = frame.copy()
                    # Add annotations for display only
                    if angle_data:
                        _, (cx, cy) = angle_data
                        rad = math.radians(current_angle)
                        dx = int(math.cos(rad) * 200); dy = int(math.sin(rad) * 200)
                        cv2.line(annotated_frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, f"Angle: {current_angle:.1f}", (cx + 10, cy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    action_text = "GAS" if gas else ("BRAKE" if brake else "COASTING")
                    cv2.putText(annotated_frame, f"Action: {action_text}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    cv2.imshow("Manual Play Recorder (Press 'q' to quit)", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.05) # ~20 FPS

        finally:
            print(f"Data saved to {OUTPUT_CSV}")
            csv_file.close()
            # --- 4. CLEAN UP WINDOWS CONDITIONALLY ---
            if not args.no_display:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()