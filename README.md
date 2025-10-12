# Hill-Climb-Racing-rl
Hobby project to play around with reinforcement learning to see if it can play classic version of Hill Climb Racing.

![Leave Temp Snip](https://github.com/bbartling/Hill-Climb-Racing-rl/blob/develop/snip.png)


## Getting Setup
* Test on Windows 10 and 11
* Python version 3.12.x
* Hill Climb Racing App downloaded from Windows App store
* Windows Terminal App downloaded from Windows App store to run PowerShell commands

Install Python packages:
```bash
pip install pyautogui mss opencv-python numpy scikit-image Pillow easyocr keyboard
```
* Hit `q` on keyboard to quit


### **Imitation Learning**
* Instead of the trial-and-error of Reinforcement Learning (RL), the AI learns by directly mimicking the actions of an expert—in this case, the human player...

The process is straightforward:

1.  **Record Data:** You play the game, and a script records what you *see* (the state, like the car's angle) and what you *do* (the action, like pressing gas or brake).
2.  **Train a Model:** You use this recorded data to train a model that learns to predict the correct action for a given state.
3.  **Deploy the AI:** A new script uses the trained model to play the game on its own.

-----

## Part 1: The Data Recording Script

Python script that records game play and angle of Jeep.

  * Capture the game screen.
  * Use your angle-detection logic to find the jeep's angle.
  * Detect if you are pressing the **left arrow key (Brake)** or the **right arrow key (Gas)**.
  * Display an `imshow` window showing the live-annotated angle and your current action.
  * Save the `timestamp`, `angle`, `gas_pressed`, and `brake_pressed` to a CSV file.


-----

## Part 2: How to Create an AI from This Data

Once you have a good amount of data (play for a few minutes, generating a few thousand rows in `manual_play_data.csv`), you can train your AI.

### The Concept: Behavioral Cloning

Your `manual_play_data.csv` is now a labeled dataset.

  * **Input Features (X):** The `angle` of the car. (You could add more features later, like slope or being airborne, just like in your RL script).
  * **Output Label (y):** The action you took (`[gas_pressed, brake_pressed]`). This is a classification problem: given the angle, should the AI press gas, brake, or nothing?

### Steps to Build the AI

1.  **Load the Data:** Use a library like `pandas` to load `manual_play_data.csv`.
2.  **Prepare the Data:** Separate your columns into features (`X`, which is just the 'angle' column) and labels (`y`, which would be the 'gas\_pressed' and 'brake\_pressed' columns). You'll want to convert the two action columns into a single "action" column (e.g., 0 for Coast, 1 for Gas, 2 for Brake).
3.  **Build a Model:** A simple neural network using **TensorFlow/Keras** or **PyTorch** is perfect for this. It would take the angle as input and have three outputs, one for each possible action (Coast, Gas, Brake).
4.  **Train the Model:** You feed your recorded data into the model (`model.fit(X, y)`). The model will learn the patterns, figuring out things like "if the angle is tilted far forward, the human usually presses the brake."
5.  **Create a Player Script:** This script would be similar to your `hcr_rl.py` or `hcr_heuristic.py`, but instead of complex logic or a Q-table, it would:
      * Get the current car angle.
      * Feed that angle to your trained model: `predicted_action = model.predict(current_angle)`.
      * Execute the action (`pyautogui.mouseDown`) that the model chose.


## TODO Reinforcment Learning