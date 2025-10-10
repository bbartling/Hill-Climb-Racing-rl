
import sys
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

# =============================
# --- CONFIGURATION & GLOBALS ---
# =============================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
EPISODES = 100
STEPS_PER_EP = 4000
LOOP_HZ = 20
ALPHA = 0.2
GAMMA = 0.95
EPS_START, EPS_END, EPS_DECAY = 0.6, 0.05, 70
CAR_RED_MIN = (180, 0, 0)
CAR_RED_MAX = (255, 100, 100)
ANGLE_OK_THRESH, ANGLE_TILT_THRESH, ANGLE_FLIP_THRESH = 1.0, 1.6, 2.2
FUEL_LOW_FRAC = 0.08
RECOGNITION_ROI = {'x1': 0.4, 'y1': 0.4, 'x2': 0.6, 'y2': 0.6}
SIMILARITY_THRESHOLD = 0.6
MOUSE_PRESS_SECONDS = 0.08

# =============================
# --- HELPER FUNCTIONS ---
# =============================

def load_config(path="hcr_config.json"):
    """Loads the config file and decodes the reference images."""
    try:
        with open(path, "r") as f:
            config = json.load(f)
        
        # Handle single-object stages
        for key in ['setup', 'menu', 'game_over']:
            if config.get(key) and config[key].get('ref_img_b64'):
                img_bytes = base64.b64decode(config[key]['ref_img_b64'])
                img = Image.open(io.BytesIO(img_bytes))
                config[key]['ref_img_np'] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # <-- MODIFIED: Special handling for the 'gameplay' list
        if 'gameplay' in config and isinstance(config['gameplay'], list) and len(config['gameplay']) > 0:
            # We'll decode the image for the first gameplay set to use as the reference
            gameplay_set = config['gameplay'][0]
            if gameplay_set.get('ref_img_b64'):
                img_bytes = base64.b64decode(gameplay_set['ref_img_b64'])
                img = Image.open(io.BytesIO(img_bytes))
                # Store the decoded image back into that first dictionary
                config['gameplay'][0]['ref_img_np'] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        else:
            print(f"❌ Error: 'gameplay' key in '{path}' is not a non-empty list. Please re-annotate.")
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
        return 1.0 - (np.mean(cv2.absdiff(img1_gray, cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0])))) / 255.0)

def get_current_screen_mode(current_frame_gray, config):
    """
    Recognizes the screen mode and returns the scores dictionary.
    """
    h, w = current_frame_gray.shape
    roi = { 'x1': int(w * RECOGNITION_ROI['x1']), 'y1': int(h * RECOGNITION_ROI['y1']),
            'x2': int(w * RECOGNITION_ROI['x2']), 'y2': int(h * RECOGNITION_ROI['y2']) }
    current_roi_gray = current_frame_gray[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
    
    scores = {}
    for mode in ['gameplay', 'menu', 'game_over']:
        if mode == 'gameplay':
            # <-- MODIFIED: Use the reference image from the first item in the gameplay list
            ref_img_np = config['gameplay'][0]['ref_img_np']
        else:
            ref_img_np = config[mode]['ref_img_np']
            
        ref_roi_gray = ref_img_np[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
        scores[mode] = get_image_similarity(current_roi_gray, ref_roi_gray)

    best_match_mode = max(scores, key=scores.get)
    
    if scores[best_match_mode] > SIMILARITY_THRESHOLD:
        return best_match_mode.upper(), scores
    return "UNKNOWN", scores

def get_car_orientation(frame_rgb):
    mask = cv2.inRange(frame_rgb, np.array(CAR_RED_MIN), np.array(CAR_RED_MAX)); rows, cols = np.where(mask)
    if len(rows) < 50: return 0
    height = np.max(rows) - np.min(rows); width = np.max(cols) - np.min(cols)
    if width < 10: return 3
    aspect_ratio = height / width
    if aspect_ratio > ANGLE_FLIP_THRESH: return 3
    if aspect_ratio > ANGLE_TILT_THRESH: return 2
    if aspect_ratio > ANGLE_OK_THRESH: return 1
    return 0
def sense_environment(gray_frame):
    h, w = gray_frame.shape
    def find_ground_y(g, x_col):
        col = g[int(h*0.55):, int(x_col)]; grad = np.abs(np.diff(col.astype(np.int16)))
        return (np.argmax(grad) + int(h*0.55)) if grad.size > 0 and np.max(grad) > 18 else int(h*0.8)
    gy1, gy2 = find_ground_y(gray_frame, w*0.58), find_ground_y(gray_frame, w*0.72)
    slope = 0 if (gy2 - gy1) > 4 else (2 if (gy2 - gy1) < -4 else 1)
    airborne = int(gray_frame[max(0, find_ground_y(gray_frame, w*0.46) - 8), int(w*0.46)] > 200)
    return slope, airborne
def get_state_index(slope, air, fuel_low, angle_code): return angle_code + (fuel_low * 4) + (air * 8) + (slope * 16)
def choose_action(Q_table, state, eps): return random.choice([0, 1, 2]) if random.random() < eps else int(np.argmax(Q_table[state]))

def perform_action(action, config, game_region):
    if action == 0: return
    pedal_key = 'gas_roi' if action == 1 else 'brake_roi'
    try:
        # <-- MODIFIED: Use the pedal ROI from the first item in the gameplay list
        roi = config['gameplay'][0][pedal_key]
        if roi is None: return
        center_x = game_region['left'] + (roi['x1'] + roi['x2']) // 2
        center_y = game_region['top'] + (roi['y1'] + roi['y2']) // 2
        pyautogui.mouseDown(center_x, center_y); time.sleep(MOUSE_PRESS_SECONDS); pyautogui.mouseUp(center_x, center_y)
    except (KeyError, IndexError): pass

def get_reward(angle_code, prev_angle_code):
    if angle_code == 3: return -20.0;
    if angle_code == 2: return -2.0;
    if angle_code == 1: return -0.5;
    if angle_code < prev_angle_code: return 2.0;
    return 0.1

# =============================
# --- MAIN EXECUTION LOOP ---
# =============================

def main():
    print("--- Hill Climb Racing RL Agent (v5 - CSV Logging) ---")
    config = load_config()
    
    if 'setup' not in config or 'game_region' not in config['setup']:
        print("\n❌ FATAL ERROR: 'game_region' not found. Please re-run the annotator script.")
        sys.exit(1)
        
    game_region = config['setup']['game_region']
    Q = np.zeros((48, 3), dtype=np.float32)
    
    log_file_path = 'training_log.csv'
    log_headers = ['episode', 'total_reward', 'steps_survived', 'epsilon']
    
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
    
    print(f"📝 Logging training progress to {log_file_path}")
    
    with mss() as sct:
        for ep in range(EPISODES):
            print(f"\n--- Episode {ep + 1}/{EPISODES} ---")
            
            episode_reward, is_episode_running = 0, False
            state, prev_angle, action = 0, 0, 0
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * ep / EPS_DECAY)
            
            final_steps = 0
            
            for step in range(STEPS_PER_EP):
                frame_rgb = np.array(sct.grab(game_region))[:, :, :3]
                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                mode, scores = get_current_screen_mode(frame_gray, config)
                
                print(f"Step {step} | 🔎 Screen scores: GP={scores.get('gameplay', 0):.2f}, M={scores.get('menu', 0):.2f}, GO={scores.get('game_over', 0):.2f} -> Match: {mode}")

                if mode == "MENU":
                    if is_episode_running: break
                    print("   -> Action: Clicking start button.")
                    roi = config['menu']['start_button_roi']
                    click_x = game_region['left'] + (roi['x1'] + roi['x2']) // 2
                    click_y = game_region['top'] + (roi['y1'] + roi['y2']) // 2
                    pyautogui.click(click_x, click_y)
                    time.sleep(1.0)
                    continue

                elif mode == "GAME_OVER":
                    print("   -> State: Game Over screen detected.")
                    
                    print("   -> Action: Clicking restart area.")
                    roi = config['game_over']['restart_area_roi']
                    click_x = game_region['left'] + (roi['x1'] + roi['x2']) // 2
                    click_y = game_region['top'] + (roi['y1'] + roi['y2']) // 2
                    pyautogui.click(click_x, click_y)
                    time.sleep(1.0)

                    if is_episode_running:
                        print("   -> Finalizing episode after crash.")
                        reward = -20.0
                        episode_reward += reward
                        Q[state, action] += ALPHA * (reward - Q[state, action])
                        final_steps = step
                        break 
                    else:
                        continue

                elif mode == "GAMEPLAY":
                    if not is_episode_running:
                        print("   -> State: Gameplay detected. Starting RL.")
                        is_episode_running = True

                    # <-- MODIFIED: Use the fuel ROI from the first item in the gameplay list
                    fuel_roi = config['gameplay'][0]['fuel_roi']
                    fuel_img = frame_rgb[fuel_roi['y1']:fuel_roi['y2'], fuel_roi['x1']:fuel_roi['x2']]
                    green_frac = np.mean((fuel_img[:,:,1] > 140) & (fuel_img[:,:,0] < 120))
                    fuel_low = int(green_frac < FUEL_LOW_FRAC)
                    slope, air = sense_environment(frame_gray); angle = get_car_orientation(frame_rgb)
                    state_next = get_state_index(slope, air, fuel_low, angle)
                    print(f"   -> Sensed State: Slope={slope}, Air={air}, FuelLow={fuel_low}, Angle={angle} -> Index={state_next}")
                    
                    if step > 0:
                        reward = get_reward(angle, prev_angle)
                        episode_reward += reward
                        best_future_q = np.max(Q[state_next])
                        q_target = reward + GAMMA * best_future_q
                        Q[state, action] += ALPHA * (q_target - Q[state, action])
                    
                    action = choose_action(Q, state_next, epsilon)
                    action_name = ["COAST", "GAS", "BRAKE"][action]
                    print(f"   -> Action: Epsilon={epsilon:.2f}, Decided={action_name}")
                    perform_action(action, config, game_region)

                    state = state_next
                    prev_angle = angle
                else:
                    print(f"   -> State: Unknown screen. Waiting...")
                    time.sleep(0.5)
                
                final_steps = step
                time.sleep(1.0 / LOOP_HZ)
            
            with open(log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep + 1, episode_reward, final_steps, epsilon])
            
            print(f"🏁 Episode {ep + 1} Reward: {episode_reward:.2f} | Steps: {final_steps}")

    print("\n--- Training complete. ---")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n🛑 Training interrupted by user.")
    except pyautogui.FailSafeException: print("\n🛑 PyAutoGUI failsafe triggered (mouse moved to a corner).")