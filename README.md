# 🏎️ Hill-Climb-Racing-RL

A fun hobby project exploring **computer vision**, **imitation learning**, and eventually **reinforcement learning (RL)** — teaching an AI to play the classic *Hill Climb Racing* game. Click on the image below to go to a Google drive video capture of game play with computer vision processing to calculate the jeeps angle and hieght.

[![Gameplay Screenshot](https://github.com/bbartling/Hill-Climb-Racing-rl/blob/develop/snip.png)](https://drive.google.com/file/d/162kejk2QqyGb1krAq7rsnFM0XDiPQKWC/view?usp=sharing)

---

## 🧩 Getting Started

* **OS:** Tested on Windows 10 or 11. Should work fine on Mac as well.
* **Python:** 3.12.x
* **Game:** *Hill Climb Racing* from the Microsoft Store
* **Terminal:** Windows Terminal App (for PowerShell commands)

Python packages:
```bash
pip install pyautogui mss opencv-python numpy scikit-image Pillow easyocr keyboard
```
---

## Computer Vision Testing

This project aims to build an AI that learns to drive like a human in *Hill Climb Racing* but first the AI needs to see data in the game.

### Test computer vision on still image screenshot of game play
```bash
python main.py --images-in config_screenshots/images_for_cv --images-out config_screenshots/procressed_images --csv images_stats.csv
```

### Test computer vision on recorded video file of game play
```bash
python main.py --video-in "C:/Users/ben/Videos/HCR/HCR_raw.mp4" --video-out "C:/Users/ben/Videos/HCR/HCR_processed.mp4" --csv "C:/Users/ben/Videos/HCR/video_stats.csv"
```

### Record data set for NN
```bash
# Example region (LEFT TOP WIDTH HEIGHT). Adjust to your monitor/game window.
python main.py --record --region 68 35 1235 687 --csv manual_play_data.csv --fps 30
```
* Press s to start, q to quit.
* Default keys: right arrow = gas, left arrow = brake
* Add --no-display for headless environments

---

## Train NN in Pytorch for imitation learning
TODO

---

## 📜 License
MIT License © 2025 Ben Bartling

