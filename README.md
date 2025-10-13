# 🏎️ Hill-Climb-Racing-RL

A fun hobby project exploring **computer vision**, **imitation learning**, and eventually **reinforcement learning (RL)** — teaching an AI to play the classic *Hill Climb Racing* game.

![Gameplay Screenshot](https://github.com/bbartling/Hill-Climb-Racing-rl/blob/develop/snip.png)

https://drive.google.com/file/d/162kejk2QqyGb1krAq7rsnFM0XDiPQKWC/view?usp=sharing
---

## 🎯 Overview

This project aims to build an AI that learns to drive like a human in *Hill Climb Racing*.

The workflow is divided into stages:

1. **Annotate** the game’s regions of interest (fuel bar, gas/brake pedals, etc.).
2. **Record** human gameplay data (state + actions).
3. **Train** a neural network via **behavioral cloning** (imitation learning).
4. **(TODO)** Extend to full **reinforcement learning (RL)** using the trained model as a baseline.

### Video processing and testing of computer vision

The script detects the jeep by isolating red hues in the image using HSV color thresholds, then finds the largest red contour that represents the jeep’s body. It fits a minimum-area rectangle around that contour to determine the jeep’s rotation and computes the angle based on the rectangle’s orientation. The centroid of that red contour serves as the reference point for height calculation. From the centroid, the script scans straight downward until it encounters pixels matching ground colors (green or brown), marking that spot as the contact point. The vertical distance in pixels between the centroid and this ground boundary is labeled on the image as the jeep’s height above ground.

```bash
$env:HCR_VIDEO = "C:\Users\ben\Videos\HCR.mp4"


python video_jeep_angle_and_height_testing.py

```

[![Watch gameplay on Bens Drive](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://drive.google.com/file/d/162kejk2QqyGb1krAq7rsnFM0XDiPQKWC/view?usp=sharing)

---

## 🧩 Getting Started

### Requirements

* **OS:** Windows 10 or 11
* **Python:** 3.12.x
* **Game:** *Hill Climb Racing* from the Microsoft Store
* **Terminal:** Windows Terminal App (for PowerShell commands)

### Installation

Install dependencies:

```bash
pip install pyautogui mss opencv-python numpy scikit-image Pillow easyocr keyboard
```

> Press **`q`** anytime in a capture window to quit safely.

---

## 🖼️ Step 1 — Annotate the Game Screen

Run the annotator to define key regions of interest:

```bash
python hcr_annotator.py
```

You’ll:

* Capture or load screenshots of your monitor.
* Draw rectangles around UI elements like the **fuel bar**, **pedals**, and **distance meter**.
* Save the configuration as `hcr_config.json`.

This step ensures consistent cropping and analysis across scripts.

---

## 🎮 Step 2 — Record Human Gameplay

Once your config is ready, record human gameplay with:

```bash
python record_manual_play.py
```

This script:

* Captures screen frames from the defined game region.
* Detects the jeep’s angle in real time.
* Logs your keyboard input (`←` brake, `→` gas).
* Saves data to a CSV file:

| timestamp    | angle | gas_pressed | brake_pressed |
| ------------ | ----- | ----------- | ------------- |
| 1697051887.1 | 78.6  | 1           | 0             |

Use:

```bash
python record_manual_play.py --no-display
```

for **headless mode** (no preview window).

---

## 🧠 Step 3 — Train an Imitation Learning Model (TODO)

With `manual_play_data.csv` ready, the next step is to train a simple neural network that mimics your driving:

### Concept — *Behavioral Cloning*

The model learns to map observed **game states** to **player actions**.

* **Input (X):** Jeep angle (and potentially more features later)
* **Output (y):** Player action

  * `0 = coast`, `1 = gas`, `2 = brake`

### Example Workflow

1. Load CSV data with `pandas`.
2. Split into `train/test` datasets.
3. Train a small neural net using **TensorFlow/Keras** or **PyTorch**:

   ```python
   model.fit(X_train, y_train, epochs=20, batch_size=64)
   ```
4. Use the trained model in a script that plays automatically:

   ```python
   action = model.predict(current_angle)
   ```

---

## 🚀 Step 4 — Reinforcement Learning (Future Work)

Planned future enhancement:

* Replace the imitation policy with an **RL agent**.
* Use rewards for fuel efficiency, distance traveled, and airtime stability.
* Explore **Deep Q-Learning (DQN)** or **PPO** frameworks.

---

## 🧭 Current Status

| Component               | Status                           |
| ----------------------- | -------------------------------- |
| `hcr_annotator.py`      | ✅ Fully functional               |
| `record_manual_play.py` | ✅ Recording human gameplay works |
| `manual_play_data.csv`  | ✅ Generates training data        |
| Imitation Learning NN   | ⏳ Next task                      |
| RL Agent                | 🧠 Future TODO                   |

---

## 💡 Notes

* This is a local, offline experimentation project — no network or cloud dependencies.
* Run each script directly in PowerShell or VS Code’s terminal.
* Data files (`.json`, `.csv`) are written in the same directory.

---

## 🏁 Roadmap

| Stage | Description                             | Status |
| ----- | --------------------------------------- | ------ |
| 1     | Build screen annotation tool            | ✅      |
| 2     | Record human gameplay dataset           | ✅      |
| 3     | Train imitation learning NN             | 🔜     |
| 4     | Add reinforcement learning agent        | 🚧     |
| 5     | Visualize performance & learning curves | 🚧     |

---

## 📜 License

MIT License © 2025 Ben Bartling

