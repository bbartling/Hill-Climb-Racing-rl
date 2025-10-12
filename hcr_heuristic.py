import sys
import os
import csv
import time
import json
import numpy as np
import pyautogui
from mss import mss
import cv2
import base64
import io
from PIL import Image
import easyocr
import keyboard

# =============================
# --- CONFIGURATION & GLOBALS ---
# =============================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
EPISODES = 10
STEPS_PER_EP = 4000
LOOP_HZ = 20
CAR_RED_MIN = (180, 0, 0)
CAR_RED_MAX = (255, 100, 100)
ANGLE_OK_THRESH, ANGLE_TILT_THRESH, ANGLE_FLIP_THRESH = 1.0, 1.6, 2.2
RECOGNITION_ROI = {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}
SIMILARITY_THRESHOLD = 0.5

# =============================
# --- HELPER FUNCTIONS ---
# =============================

_EASY_READER = None
def _get_easy_reader():
    global _EASY_READER
    if _EASY_READER is None:
        _EASY_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASY_READER

def ocr_meters_from_roi(frame_rgb, roi):
    if roi is None: return None
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0: return None
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = _get_easy_reader().readtext(crop, detail=0, paragraph=False)
    for text in results:
        digits = "".join([c for c in text if c.isdigit()])
        if digits:
            try: return int(digits)
            except ValueError: pass
    return None

def load_config(path="hcr_config.json"):
    try:
        with open(path, "r") as f: config = json.load(f)
        for key in ["setup", "menu", "game_over"]:
            if config.get(key) and config[key].get("ref_img_b64"):
                img_bytes = base64.b64decode(config[key]["ref_img_b64"])
                config[key]["ref_img_np"] = cv2.cvtColor(np.array(Image.open(io.BytesIO(img_bytes))), cv2.COLOR_RGB2GRAY)
        if "gameplay" in config and isinstance(config["gameplay"], list) and config["gameplay"]:
            if config["gameplay"][0].get("ref_img_b64"):
                img_bytes = base64.b64decode(config["gameplay"][0]["ref_img_b64"])
                config["gameplay"][0]["ref_img_np"] = cv2.cvtColor(np.array(Image.open(io.BytesIO(img_bytes))), cv2.COLOR_RGB2GRAY)
        else:
            print(f"❌ Error: 'gameplay' key in '{path}' is invalid."); sys.exit(1)
        print(f"✅ Configuration loaded from '{path}'"); return config
    except Exception as e:
        print(f"❌ Error loading config '{path}': {e}."); sys.exit(1)

def get_image_similarity(img1_gray, img2_gray):
    try:
        from skimage.metrics import structural_similarity as ssim
        h, w = img1_gray.shape
        img2_resized = cv2.resize(img2_gray, (w, h))
        return ssim(img1_gray, img2_resized, data_range=255)
    except ImportError:
        return 1.0 - (np.mean(cv2.absdiff(img1_gray, cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0])))) / 255.0)

def get_current_screen_mode(current_frame_gray, config):
    h, w = current_frame_gray.shape
    roi = {"x1": int(w * RECOGNITION_ROI["x1"]), "y1": int(h * RECOGNITION_ROI["y1"]), "x2": int(w * RECOGNITION_ROI["x2"]), "y2": int(h * RECOGNITION_ROI["y2"])}
    current_roi_gray = current_frame_gray[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
    scores = {}
    for mode in ["gameplay", "menu", "game_over"]:
        ref_img_np = config["gameplay"][0]["ref_img_np"] if mode == "gameplay" else config[mode]["ref_img_np"]
        ref_roi_gray = ref_img_np[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        scores[mode] = get_image_similarity(current_roi_gray, ref_roi_gray)
    best_match_mode = max(scores, key=scores.get)
    return (best_match_mode.upper(), scores) if scores[best_match_mode] > SIMILARITY_THRESHOLD else ("UNKNOWN", scores)

def get_car_orientation(frame_rgb):
    mask = cv2.inRange(frame_rgb, np.array(CAR_RED_MIN), np.array(CAR_RED_MAX))
    rows, _ = np.where(mask)
    if len(rows) < 50: return 0
    height, width = np.ptp(rows), np.ptp(np.where(mask)[1])
    if width < 10: return 3
    aspect_ratio = height / width
    if aspect_ratio > ANGLE_FLIP_THRESH: return 3
    if aspect_ratio > ANGLE_TILT_THRESH: return 2
    if aspect_ratio > ANGLE_OK_THRESH: return 1
    return 0

def update_action(new_action, current_action, config, game_region):
    if new_action == current_action: return
    if current_action in [1, 2]:
        pedal_key = "gas_roi" if current_action == 1 else "brake_roi"
        try:
            roi = config["gameplay"][0][pedal_key]
            pyautogui.mouseUp(game_region["left"] + (roi["x1"] + roi["x2"]) // 2, game_region["top"] + (roi["y1"] + roi["y2"]) // 2)
        except (KeyError, IndexError): pyautogui.mouseUp()
    if new_action in [1, 2]:
        pedal_key = "gas_roi" if new_action == 1 else "brake_roi"
        try:
            roi = config["gameplay"][0][pedal_key]
            pyautogui.mouseDown(game_region["left"] + (roi["x1"] + roi["x2"]) // 2, game_region["top"] + (roi["y1"] + roi["y2"]) // 2)
        except (KeyError, IndexError): pass

def get_frame(game_region):
    with mss() as sct:
        monitor = {"left": game_region["left"], "top": game_region["top"], "width": game_region["width"], "height": game_region["height"]}
        sct_img = sct.grab(monitor)
        frame_rgb = cv2.cvtColor(np.array(sct_img)[..., :3], cv2.COLOR_BGR2RGB)
        return frame_rgb, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

def detect_mode(frame_rgb, config):
    return get_current_screen_mode(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), config)[0]

def start_game_from_menu(config, game_region):
    try:
        roi = config["menu"]["start_button_roi"]
        pyautogui.click(game_region["left"] + (roi["x1"] + roi["x2"]) // 2, game_region["top"] + (roi["y1"] + roi["y2"]) // 2)
    except Exception:
        pyautogui.click(game_region["left"] + game_region["width"] // 2, game_region["top"] + game_region["height"] // 2)
    time.sleep(0.2)


# =========================================================
# --- REVISED SENSING AND HEURISTIC LOGIC ---
# =========================================================
def sense_and_decide(frame_gray, frame_rgb):
    """
    Senses the environment (airborne, angle, slope) and decides the best action.
    - Action 1: Hold Gas
    - Action 2: Hold Brake
    Returns the chosen action and a dictionary of the current state for logging.
    """
    h, w = frame_gray.shape

    # --- 1. Sense Ground and Slope ---
    def find_ground_y(g, x_col):
        col = g[int(h * 0.55):, int(x_col)]
        grad = np.abs(np.diff(col.astype(np.int16)))
        return (np.argmax(grad) + int(h * 0.55)) if grad.size > 0 and np.max(grad) > 18 else int(h * 0.8)

    ground_y_near = find_ground_y(frame_gray, w * 0.58)
    ground_y_far = find_ground_y(frame_gray, w * 0.72)

    slope_diff = ground_y_near - ground_y_far
    if slope_diff < -8: slope = 2  # Steep Uphill
    elif slope_diff < -2: slope = 1 # Uphill
    else: slope = 0 # Flat/Downhill

    # --- 2. Sense if Airborne ---
    airborne = frame_gray[max(0, find_ground_y(frame_gray, w * 0.46) - 8), int(w * 0.46)] > 200

    # --- 3. Sense Car Orientation ---
    angle_code = get_car_orientation(frame_rgb)

    # --- 4. Heuristic Decision Logic ---
    action = 1 # Default to Gas
    if not airborne:
        action = 1 # On the ground, always hold gas.
    else: # In the air, control rotation.
        if angle_code > 1: action = 2 # Tilted badly, use BRAKE to level out.
        elif slope == 2: action = 2 # Flying towards a steep wall, use BRAKE to soften landing.
        else: action = 1 # Reasonably level, use GAS to prepare for landing.

    state_info = {"airborne": airborne, "angle": angle_code, "slope": slope}
    return action, state_info

# =============================
# --- MAIN EXECUTION LOOP ---
# =============================
def main():
    config = load_config()
    _get_easy_reader()
    game_region = config["setup"]["game_region"]
    csv_path = "heuristic_log.csv"

    log_headers = ["episode", "steps_survived", "final_distance_m"]
    file_exists = os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if not file_exists: writer.writerow(log_headers)

    go_dist_roi = config.get("game_over", {}).get("distance_meter_roi")

    try:
        for ep in range(EPISODES):
            print(f"\n===== EPISODE {ep+1}/{EPISODES} (Heuristic Strategy) =====")
            active_action = 0
            final_steps = 0
            episode_final_distance = 0

            for step in range(STEPS_PER_EP):
                loop_start_time = time.time()

                if keyboard.is_pressed('q'):
                    print("\n'q' key pressed. Stopping...")
                    raise KeyboardInterrupt

                frame_rgb, frame_gray = get_frame(game_region)
                mode = detect_mode(frame_rgb, config)

                if mode == "MENU":
                    start_game_from_menu(config, game_region)
                    time.sleep(0.5)
                    continue

                elif mode == "GAMEPLAY":
                    # --- Sense, Decide, and Log ---
                    new_action, state = sense_and_decide(frame_gray, frame_rgb)
                    action_str = "Gas" if new_action == 1 else "Brake"
                    print(f"   -> State: Airborne={state['airborne']}, Angle={state['angle']}, Slope={state['slope']} | Action: {action_str}", end='\r')

                    # --- Act ---
                    update_action(new_action, active_action, config, game_region)
                    active_action = new_action
                    final_steps = step

                elif mode == "GAME_OVER":
                    print("\n   -> State: Game Over screen detected.") # Newline to clear the state log
                    dist = ocr_meters_from_roi(frame_rgb, go_dist_roi)
                    if dist is not None:
                        episode_final_distance = dist
                        print(f"   -> Final distance from OCR: {dist} m")

                    pyautogui.click(game_region["left"] + game_region["width"] // 2, game_region["top"] + game_region["height"] // 2)
                    time.sleep(1.0)
                    break

                else:
                    print("   -> Unknown mode; waiting...", end="\r")
                    time.sleep(0.1)

                # Maintain loop frequency
                elapsed = time.time() - loop_start_time
                if (1.0 / LOOP_HZ) > elapsed:
                    time.sleep((1.0 / LOOP_HZ) - elapsed)

            # --- End of episode ---
            writer.writerow([ep + 1, final_steps, episode_final_distance])
            csv_f.flush()
            print(f"EP {ep+1} summary: steps={final_steps}, final_distance_m={episode_final_distance}")

    finally:
        print("\nClosing log file.")
        if csv_f:
            csv_f.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except pyautogui.FailSafeException:
        print("\n🛑 PyAutoGUI failsafe triggered (mouse moved to a corner).")
    finally:
        print("Releasing any pressed mouse buttons...")
        pyautogui.mouseUp()
        print("✅ Script finished.")