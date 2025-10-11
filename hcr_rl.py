import sys
import os
import csv
import time
import random
import json
import numpy as np
import pyautogui
from mss import mss
import cv2
import base64
import io
from PIL import Image
import easyocr

# =============================
# --- CONFIGURATION & GLOBALS ---
# =============================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
EPISODES = 5
STEPS_PER_EP = 4000
LOOP_HZ = 20
ALPHA = 0.2
GAMMA = 0.95
EPS_START, EPS_END, EPS_DECAY = 0.6, 0.05, 70
CAR_RED_MIN = (180, 0, 0)
CAR_RED_MAX = (255, 100, 100)
ANGLE_OK_THRESH, ANGLE_TILT_THRESH, ANGLE_FLIP_THRESH = 1.0, 1.6, 2.2
FUEL_LOW_FRAC = 0.08
RECOGNITION_ROI = {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}
SIMILARITY_THRESHOLD = 0.5
MOUSE_PRESS_SECONDS = 0.08

# =============================
# --- HELPER FUNCTIONS ---
# =============================

# ---------- EasyOCR setup ----------
_EASY_READER = None
def _get_easy_reader():
    global _EASY_READER
    if _EASY_READER is None:
        # GPU False is fine and avoids CUDA issues
        _EASY_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASY_READER

def ocr_meters_from_roi(frame_rgb, roi):
    """
    Read '##m' text from a HUD region using EasyOCR.
    Returns an integer meters value, or None if not found.
    """
    if roi is None: 
        return None
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Light preproc: scale up improves OCR accuracy
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    reader = _get_easy_reader()
    # detail=0 -> strings only; paragraph=False avoids joining with newlines
    results = reader.readtext(crop, detail=0, paragraph=False)

    # Parse something like '14m' / '133m'
    for text in results:
        if not text:
            continue
        # keep digits from the string; expect a trailing 'm'
        digits = "".join([c for c in text if c.isdigit()])
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return None

def load_config(path="hcr_config.json"):
    """Loads the config file and decodes the reference images."""
    try:
        with open(path, "r") as f:
            config = json.load(f)

        # Handle single-object stages
        for key in ["setup", "menu", "game_over"]:
            if config.get(key) and config[key].get("ref_img_b64"):
                img_bytes = base64.b64decode(config[key]["ref_img_b64"])
                img = Image.open(io.BytesIO(img_bytes))
                config[key]["ref_img_np"] = cv2.cvtColor(
                    np.array(img), cv2.COLOR_RGB2GRAY
                )

        # <-- MODIFIED: Special handling for the 'gameplay' list
        if (
            "gameplay" in config
            and isinstance(config["gameplay"], list)
            and len(config["gameplay"]) > 0
        ):
            # We'll decode the image for the first gameplay set to use as the reference
            gameplay_set = config["gameplay"][0]
            if gameplay_set.get("ref_img_b64"):
                img_bytes = base64.b64decode(gameplay_set["ref_img_b64"])
                img = Image.open(io.BytesIO(img_bytes))
                # Store the decoded image back into that first dictionary
                config["gameplay"][0]["ref_img_np"] = cv2.cvtColor(
                    np.array(img), cv2.COLOR_RGB2GRAY
                )
        else:
            print(
                f"❌ Error: 'gameplay' key in '{path}' is not a non-empty list. Please re-annotate."
            )
            sys.exit(1)

        print(f"✅ Configuration and reference images loaded from '{path}'")
        return config
    except Exception as e:
        print(f"❌ Error loading config '{path}': {e}.")
        sys.exit(1)


def get_image_similarity(img1_gray, img2_gray):
    """Compares two grayscale images using Structural Similarity Index (SSIM)."""
    try:
        from skimage.metrics import structural_similarity as ssim

        h, w = img1_gray.shape
        img2_resized = cv2.resize(img2_gray, (w, h))
        return ssim(img1_gray, img2_resized, data_range=255)
    except ImportError:
        return 1.0 - (
            np.mean(
                cv2.absdiff(
                    img1_gray,
                    cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0])),
                )
            )
            / 255.0
        )


def get_current_screen_mode(current_frame_gray, config):
    """
    Recognizes the screen mode and returns the scores dictionary.
    """
    h, w = current_frame_gray.shape
    roi = {
        "x1": int(w * RECOGNITION_ROI["x1"]),
        "y1": int(h * RECOGNITION_ROI["y1"]),
        "x2": int(w * RECOGNITION_ROI["x2"]),
        "y2": int(h * RECOGNITION_ROI["y2"]),
    }
    current_roi_gray = current_frame_gray[roi["y1"] : roi["y2"], roi["x1"] : roi["x2"]]

    scores = {}
    for mode in ["gameplay", "menu", "game_over"]:
        if mode == "gameplay":
            # <-- MODIFIED: Use the reference image from the first item in the gameplay list
            ref_img_np = config["gameplay"][0]["ref_img_np"]
        else:
            ref_img_np = config[mode]["ref_img_np"]

        ref_roi_gray = ref_img_np[roi["y1"] : roi["y2"], roi["x1"] : roi["x2"]]
        scores[mode] = get_image_similarity(current_roi_gray, ref_roi_gray)

    best_match_mode = max(scores, key=scores.get)

    if scores[best_match_mode] > SIMILARITY_THRESHOLD:
        return best_match_mode.upper(), scores
    return "UNKNOWN", scores


def get_car_orientation(frame_rgb):
    mask = cv2.inRange(frame_rgb, np.array(CAR_RED_MIN), np.array(CAR_RED_MAX))
    rows, cols = np.where(mask)
    if len(rows) < 50:
        return 0
    height = np.max(rows) - np.min(rows)
    width = np.max(cols) - np.min(cols)
    if width < 10:
        return 3
    aspect_ratio = height / width
    if aspect_ratio > ANGLE_FLIP_THRESH:
        return 3
    if aspect_ratio > ANGLE_TILT_THRESH:
        return 2
    if aspect_ratio > ANGLE_OK_THRESH:
        return 1
    return 0


def sense_environment(gray_frame):
    h, w = gray_frame.shape

    def find_ground_y(g, x_col):
        col = g[int(h * 0.55) :, int(x_col)]
        grad = np.abs(np.diff(col.astype(np.int16)))
        return (
            (np.argmax(grad) + int(h * 0.55))
            if grad.size > 0 and np.max(grad) > 18
            else int(h * 0.8)
        )

    gy1, gy2 = find_ground_y(gray_frame, w * 0.58), find_ground_y(gray_frame, w * 0.72)
    slope = 0 if (gy2 - gy1) > 4 else (2 if (gy2 - gy1) < -4 else 1)
    airborne = int(
        gray_frame[max(0, find_ground_y(gray_frame, w * 0.46) - 8), int(w * 0.46)] > 200
    )
    return slope, airborne


def get_state_index(slope, air, fuel_low, angle_code):
    return angle_code + (fuel_low * 4) + (air * 8) + (slope * 16)


def choose_action(Q_table, state, eps):
    return (
        random.choice([0, 1, 2])
        if random.random() < eps
        else int(np.argmax(Q_table[state]))
    )


def perform_action(action, config, game_region):
    if action == 0:
        return
    pedal_key = "gas_roi" if action == 1 else "brake_roi"
    try:
        # <-- MODIFIED: Use the pedal ROI from the first item in the gameplay list
        roi = config["gameplay"][0][pedal_key]
        if roi is None:
            return
        center_x = game_region["left"] + (roi["x1"] + roi["x2"]) // 2
        center_y = game_region["top"] + (roi["y1"] + roi["y2"]) // 2
        pyautogui.mouseDown(center_x, center_y)
        time.sleep(MOUSE_PRESS_SECONDS)
        pyautogui.mouseUp(center_x, center_y)
    except (KeyError, IndexError):
        pass


def get_reward(angle_code, prev_angle_code):
    if angle_code == 3:
        return -20.0
    if angle_code == 2:
        return -2.0
    if angle_code == 1:
        return -0.5
    if angle_code < prev_angle_code:
        return 2.0
    return 0.1


def get_frame(game_region):
    """Capture the game region using MSS and return (RGB, GRAY) numpy arrays."""
    with mss() as sct:
        monitor = {
            "left": game_region["left"],
            "top": game_region["top"],
            "width": game_region["width"],
            "height": game_region["height"],
        }
        sct_img = sct.grab(monitor)
        frame_rgb = np.array(sct_img)[..., :3]  # drop alpha
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return frame_rgb, frame_gray


def detect_mode(frame_rgb, config):
    """Wrapper that converts RGB to gray and calls get_current_screen_mode()."""
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mode, _ = get_current_screen_mode(frame_gray, config)
    return mode


def start_game_from_menu(config, game_region):
    """Clicks the Start button on the MENU screen using the ROI from config."""
    try:
        roi = config["menu"]["start_button_roi"]
        cx = game_region["left"] + (roi["x1"] + roi["x2"]) // 2
        cy = game_region["top"]  + (roi["y1"] + roi["y2"]) // 2
    except Exception:
        # Fallback: center click
        cx = game_region["left"] + game_region["width"] // 2
        cy = game_region["top"]  + game_region["height"] // 2
    pyautogui.click(cx, cy)
    time.sleep(0.2)

def is_fuel_low(frame_rgb, config):
    """
    Determines if the fuel gauge is low based on pixel brightness in the fuel ROI.
    Returns True if the fuel fraction (bright pixels) is below FUEL_LOW_FRAC.
    """
    fuel_roi = config["gameplay"][0]["fuel_roi"]  # will KeyError if missing (as desired)
    x1, y1, x2, y2 = fuel_roi["x1"], fuel_roi["y1"], fuel_roi["x2"], fuel_roi["y2"]
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("fuel_roi crop is empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    bright_frac = np.mean((gray / 255.0) > 0.6)
    fuel_low = bright_frac < FUEL_LOW_FRAC
    print(f"   -> Fuel gauge bright fraction={bright_frac:.3f} | low={fuel_low}")
    return fuel_low


def main():
    # ----------------- setup & logging -----------------
    config = load_config()
    _get_easy_reader() 

    game_region = config["setup"]["game_region"]
    ep_count = EPISODES                         # from your constants
    csv_path = "training_log.csv"

    Q = np.zeros((32, 3), dtype=np.float32)

    # extend CSV headers to include final distance
    log_headers = ["episode", "total_reward", "steps_survived", "epsilon", "final_distance_m"]
    file_exists = os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if not file_exists:
        writer.writerow(log_headers)

    # handy ROIs (optional keys; we check existence every time)
    gameplay_sets = config.get("gameplay", [])
    gameplay_dist_roi = None
    if gameplay_sets and isinstance(gameplay_sets[0], dict):
        gameplay_dist_roi = gameplay_sets[0].get("distance_roi")

    go_dist_roi = config.get("game_over", {}).get("distance_meter_roi")

    epsilon = EPS_START

    # ----------------- episodes loop -------------------
    for ep in range(ep_count):
        print(f"\n===== EPISODE {ep+1}/{ep_count} =====")
        is_episode_running = False
        episode_reward = 0.0
        episode_final_distance = 0
        prev_distance = None
        prev_angle = None
        final_steps = 0

        # (optional) ensure we’re at MENU at the start; your code may already do this
        # navigate_to_menu_if_needed(config, game_region)

        # --------------- per-step loop -----------------
        for step in range(STEPS_PER_EP):
            # capture current frame & detect mode
            frame_rgb, frame_gray = get_frame(game_region)   # your existing capture
            mode = detect_mode(frame_rgb, config)                 # returns: "MENU"|"GAMEPLAY"|"GAME_OVER"

            # ===== MENU =====
            if mode == "MENU":
                # your existing logic to click START, etc.
                # keep minimal prints to avoid spam
                start_game_from_menu(config, game_region)    # your helper that clicks start button
                is_episode_running = True
                time.sleep(0.2)
                continue

            # ===== GAMEPLAY =====
            elif mode == "GAMEPLAY":
                if not is_episode_running:
                    is_episode_running = True

                # --- OCR distance every step (console log every attempt) ---
                curr_distance = None

                curr_distance = ocr_meters_from_roi(frame_rgb, gameplay_dist_roi)
                if curr_distance is not None:
                    episode_final_distance = curr_distance
                    print(f"   -> OCR distance (gameplay): {curr_distance} m")
                else:
                    print("   -> OCR distance (gameplay): None")


                # --- sense environment & build next state (your existing logic) ---
                slope, in_air = sense_environment(frame_gray)       # your signal extraction
                angle = get_car_orientation(frame_rgb)               # your orientation feature
                fuel_low = is_fuel_low(frame_rgb, config)
                state_next = get_state_index(slope, in_air, fuel_low, angle)

                # --- reward + Q update (with distance shaping) ---
                if step > 0 and prev_angle is not None:
                    # base reward (your function)
                    reward = get_reward(angle, prev_angle)

                    # distance shaping: positive Δmeters adds a small bonus
                    if (curr_distance is not None) and (prev_distance is not None):
                        delta_d = max(0, curr_distance - prev_distance)
                        if delta_d > 0:
                            bonus = 0.05 * delta_d
                            reward += bonus
                            print(f"   -> Reward shaping: +{bonus:.3f} for Δdistance={delta_d} m")

                    episode_reward += reward

                    # Q-learning update (standard)
                    best_future_q = np.max(Q[state_next])
                    q_target = reward + GAMMA * best_future_q
                    Q[state, action] += ALPHA * (q_target - Q[state, action])

                # --- action selection + execution ---
                action = choose_action(Q, state_next, epsilon)       # your epsilon-greedy
                perform_action(action, config, game_region)          # your actuator (keys, etc.)

                # --- bookkeeping for next step ---
                state = state_next
                prev_angle = angle
                if curr_distance is not None:
                    prev_distance = curr_distance

                final_steps = step

            # ===== GAME OVER =====
            elif mode == "GAME_OVER":
                print("   -> State: Game Over screen detected.")

                # OCR readout (optional)
                go_dist = ocr_meters_from_roi(frame_rgb, go_dist_roi)
                print(f"   -> OCR distance (game over): {go_dist} m" if go_dist is not None else "   -> OCR distance (game over): None")
                if go_dist is not None:
                    episode_final_distance = go_dist

                # Click center — and press space as a backup trigger (some games need focus)
                click_x = game_region["left"] + game_region["width"] // 2
                click_y = game_region["top"]  + game_region["height"] // 2
                pyautogui.moveTo(click_x, click_y)
                pyautogui.click()
                pyautogui.press("space")  # backup key to restart if click fails
                print("   -> Clicked + pressed SPACE to restart game.")
                time.sleep(1.0)

                # Apply terminal reward and reset loop
                if is_episode_running:
                    print("   -> Finalizing episode after crash.")
                    reward = -20.0
                    episode_reward += reward
                    Q[state, action] += ALPHA * (reward - Q[state, action])
                    break
                else:
                    continue



            else:
                # Unknown mode; small delay to avoid hot loop
                print("   -> Unknown mode; waiting...")
                time.sleep(0.1)

        # --------------- end of episode ----------------
        # epsilon decay (your schedule)
        eps_range = max(1, EPS_DECAY)
        epsilon = EPS_END + (EPS_START - EPS_END) * max(0.0, (eps_range - (ep+1)) / eps_range)

        writer.writerow([ep + 1, episode_reward, final_steps, epsilon, episode_final_distance])
        csv_f.flush()
        print(f"EP {ep+1} summary: reward={episode_reward:.2f}, steps={final_steps}, "
              f"eps={epsilon:.3f}, final_distance_m={episode_final_distance}")

    csv_f.close()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except pyautogui.FailSafeException:
        print("\n🛑 PyAutoGUI failsafe triggered (mouse moved to a corner).")
