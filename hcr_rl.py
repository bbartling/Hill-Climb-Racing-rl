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
import keyboard  # <-- 1. ADDED for the kill switch

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
# MOUSE_PRESS_SECONDS is no longer needed as we now hold the pedals.

# =============================
# --- HELPER FUNCTIONS ---
# =============================

# ---------- EasyOCR setup ----------
_EASY_READER = None
def _get_easy_reader():
    global _EASY_READER
    if _EASY_READER is None:
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
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    reader = _get_easy_reader()
    results = reader.readtext(crop, detail=0, paragraph=False)
    for text in results:
        if not text:
            continue
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
        for key in ["setup", "menu", "game_over"]:
            if config.get(key) and config[key].get("ref_img_b64"):
                img_bytes = base64.b64decode(config[key]["ref_img_b64"])
                img = Image.open(io.BytesIO(img_bytes))
                config[key]["ref_img_np"] = cv2.cvtColor(
                    np.array(img), cv2.COLOR_RGB2GRAY
                )
        if (
            "gameplay" in config
            and isinstance(config["gameplay"], list)
            and len(config["gameplay"]) > 0
        ):
            gameplay_set = config["gameplay"][0]
            if gameplay_set.get("ref_img_b64"):
                img_bytes = base64.b64decode(gameplay_set["ref_img_b64"])
                img = Image.open(io.BytesIO(img_bytes))
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
    if len(rows) < 50: return 0
    height = np.max(rows) - np.min(rows)
    width = np.max(cols) - np.min(cols)
    if width < 10: return 3
    aspect_ratio = height / width
    if aspect_ratio > ANGLE_FLIP_THRESH: return 3
    if aspect_ratio > ANGLE_TILT_THRESH: return 2
    if aspect_ratio > ANGLE_OK_THRESH: return 1
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
    # Action space is now: 0=Nothing, 1=Hold Gas, 2=Hold Brake
    return (
        random.choice([0, 1, 2])
        if random.random() < eps
        else int(np.argmax(Q_table[state]))
    )

# --- 2. REWRITTEN ACTION FUNCTION ---
def update_action(new_action, current_action, config, game_region):
    """
    Manages holding and releasing pedals.
    - If new_action is the same as current_action, do nothing.
    - If switching pedals, release the old one and press the new one.
    - If new_action is 0, release any active pedal.
    """
    if new_action == current_action:
        return  # Keep doing the same thing (e.g., holding gas)

    # Release the previous pedal if one was active
    if current_action in [1, 2]:
        pedal_key = "gas_roi" if current_action == 1 else "brake_roi"
        try:
            roi = config["gameplay"][0][pedal_key]
            center_x = game_region["left"] + (roi["x1"] + roi["x2"]) // 2
            center_y = game_region["top"] + (roi["y1"] + roi["y2"]) // 2
            pyautogui.mouseUp(center_x, center_y)
        except (KeyError, IndexError):
            pyautogui.mouseUp() # Fallback release

    # Press the new pedal if a new one is chosen
    if new_action in [1, 2]:
        pedal_key = "gas_roi" if new_action == 1 else "brake_roi"
        try:
            roi = config["gameplay"][0][pedal_key]
            center_x = game_region["left"] + (roi["x1"] + roi["x2"]) // 2
            center_y = game_region["top"] + (roi["y1"] + roi["y2"]) // 2
            pyautogui.mouseDown(center_x, center_y)
        except (KeyError, IndexError):
            pass # Fail silently if pedal ROI isn't configured


def get_reward(angle_code, prev_angle_code):
    if angle_code == 3: return -20.0  # Flipped over (terminal)
    if angle_code == 2: return -2.0   # Tilted badly
    if angle_code == 1: return -0.5   # Slightly tilted
    if angle_code < prev_angle_code: return 2.0  # Recovered to a better angle
    return 0.1  # Survived another step


def get_frame(game_region):
    with mss() as sct:
        monitor = { "left": game_region["left"], "top": game_region["top"], "width": game_region["width"], "height": game_region["height"] }
        sct_img = sct.grab(monitor)
        frame_rgb = np.array(sct_img)[..., :3]
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return frame_rgb, frame_gray


def detect_mode(frame_rgb, config):
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mode, _ = get_current_screen_mode(frame_gray, config)
    return mode


def start_game_from_menu(config, game_region):
    try:
        roi = config["menu"]["start_button_roi"]
        cx = game_region["left"] + (roi["x1"] + roi["x2"]) // 2
        cy = game_region["top"]  + (roi["y1"] + roi["y2"]) // 2
    except Exception:
        cx = game_region["left"] + game_region["width"] // 2
        cy = game_region["top"]  + game_region["height"] // 2
    pyautogui.click(cx, cy)
    time.sleep(0.2)


def is_fuel_low(frame_rgb, config):
    try:
        fuel_roi = config["gameplay"][0]["fuel_roi"]
        x1, y1, x2, y2 = fuel_roi["x1"], fuel_roi["y1"], fuel_roi["x2"], fuel_roi["y2"]
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0: return True
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        bright_frac = np.mean((gray / 255.0) > 0.6)
        return bright_frac < FUEL_LOW_FRAC
    except (KeyError, IndexError, ValueError):
        return False # Assume fuel is not low if ROI is missing


def main():
    config = load_config()
    _get_easy_reader()
    game_region = config["setup"]["game_region"]
    csv_path = "training_log.csv"
    Q = np.zeros((48, 3), dtype=np.float32)

    log_headers = ["episode", "total_reward", "steps_survived", "epsilon", "final_distance_m"]
    file_exists = os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if not file_exists:
        writer.writerow(log_headers)

    go_dist_roi = config.get("game_over", {}).get("distance_meter_roi")
    epsilon = EPS_START

    try:
        for ep in range(EPISODES):
            print(f"\n===== EPISODE {ep+1}/{EPISODES} =====")
            is_episode_running = False
            episode_reward = 0.0
            episode_final_distance = 0
            prev_angle = None
            final_steps = 0
            state = 0
            active_action = 0  # 0=None, 1=Gas, 2=Brake

            for step in range(STEPS_PER_EP):
                loop_start_time = time.time()

                # --- 3. KILL SWITCH CHECK ---
                if keyboard.is_pressed('q'):
                    print("\n'q' key pressed. Stopping training...")
                    raise KeyboardInterrupt

                frame_rgb, frame_gray = get_frame(game_region)
                mode = detect_mode(frame_rgb, config)

                if mode == "MENU":
                    start_game_from_menu(config, game_region)
                    is_episode_running = True
                    time.sleep(0.2)
                    continue

                elif mode == "GAMEPLAY":
                    if not is_episode_running:
                        is_episode_running = True

                    # --- Sense environment & build next state ---
                    slope, in_air = sense_environment(frame_gray)
                    angle = get_car_orientation(frame_rgb)
                    fuel_low = is_fuel_low(frame_rgb, config)
                    state_next = get_state_index(slope, in_air, fuel_low, angle)

                    # --- Reward + Q update ---
                    if step > 0 and prev_angle is not None:
                        reward = get_reward(angle, prev_angle)
                        episode_reward += reward

                        # Q-learning update
                        best_future_q = np.max(Q[state_next])
                        q_target = reward + GAMMA * best_future_q
                        # Update Q-value for the action we TOOK in the previous state
                        Q[state, active_action] += ALPHA * (q_target - Q[state, active_action])

                    # --- Action selection + execution ---
                    new_action = choose_action(Q, state_next, epsilon)
                    update_action(new_action, active_action, config, game_region)

                    # --- Bookkeeping for next step ---
                    state = state_next
                    active_action = new_action
                    prev_angle = angle
                    final_steps = step

                elif mode == "GAME_OVER":
                    print("   -> State: Game Over screen detected.")

                    # Click center to restart
                    click_x = game_region["left"] + game_region["width"] // 2
                    click_y = game_region["top"]  + game_region["height"] // 2
                    pyautogui.click(click_x, click_y)
                    pyautogui.press("space")  # Backup key to restart if click fails
                    print("   -> Clicked + pressed SPACE to restart game.")
                    time.sleep(1.0) # Give the game time to transition

                    # --- REVISED LOGIC ---
                    # If the episode was actually running, apply the final penalty and update Q-table.
                    # Then, always break out of the step loop to end the episode.
                    if is_episode_running:
                        print("   -> Finalizing episode after crash.")
                        reward = -20.0 # Terminal penalty
                        episode_reward += reward
                        # Update Q-table for the last state and action before the crash
                        if 'state' in locals() and 'action' in locals():
                            Q[state, action] += ALPHA * (reward - Q[state, action])

                    # Always break the loop to move to the next episode.
                    # This fixes the bug where it gets stuck on the Game Over screen.
                    break

                else:
                    print("   -> Unknown mode; waiting...", end="\r")
                    time.sleep(0.1)

                # Maintain loop frequency
                elapsed = time.time() - loop_start_time
                sleep_time = (1.0 / LOOP_HZ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # --- End of episode ---
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * (ep + 1) / EPS_DECAY)

            writer.writerow([ep + 1, round(episode_reward, 2), final_steps, round(epsilon, 3), episode_final_distance])
            csv_f.flush()
            print(
                f"EP {ep+1} summary: reward={episode_reward:.2f}, steps={final_steps}, "
                f"eps={epsilon:.3f}, final_distance_m={episode_final_distance}"
            )

    finally:
        # This ensures the log file is closed even if the script crashes
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
        # --- 5. SAFETY: ALWAYS RELEASE MOUSE ON EXIT ---
        print("Releasing any pressed mouse buttons...")
        pyautogui.mouseUp()
        print("✅ Script finished.")