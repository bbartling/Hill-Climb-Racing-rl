import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import base64, io, os, time
from mss import mss
import json


# -------------------------------
# Monitor picker (unchanged)
# -------------------------------
class MonitorSelector(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Monitor to Capture")
        self.geometry("800x400")
        self.transient(parent)
        self.grab_set()
        self.selected_monitor = None
        self.preview_images = []

        tk.Label(
            self, text="Click the monitor you want to capture.", font=("Segoe UI", 14)
        ).pack(pady=10)
        self.monitor_frame = tk.Frame(self)
        self.monitor_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.populate_monitors()

    def populate_monitors(self):
        with mss() as sct:
            for i, monitor in enumerate(sct.monitors[1:]):
                frame = tk.Frame(self.monitor_frame, borderwidth=2, relief="groove")
                frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=5)

                img_sct = sct.grab(monitor)
                img_pil = Image.frombytes(
                    "RGB", img_sct.size, img_sct.bgra, "raw", "BGRX"
                )
                img_pil.thumbnail((320, 180))
                img_tk = ImageTk.PhotoImage(img_pil)
                self.preview_images.append(img_tk)

                tk.Button(
                    frame,
                    image=img_tk,
                    command=lambda m=monitor: self.select_monitor(m),
                ).pack(padx=5, pady=5)
                tk.Label(frame, text=f"Monitor {i+1}", font=("Segoe UI", 10)).pack(
                    pady=5
                )

    def select_monitor(self, monitor_details):
        self.selected_monitor = monitor_details
        self.destroy()


# -------------------------------
# Main Annotator (Simplified and Fixed)
# -------------------------------
class MultiScreenAnnotator(tk.Tk):
    STAGES = ["SETUP", "GAMEPLAY", "MENU", "GAME_OVER"]
    GAMEPLAY_TOOLS = ["fuel", "gas", "brake", "distance"]

    def __init__(self):
        super().__init__()
        self.title("HCR - Annotator (Simplified)")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.current_img = None
        self.current_tk_img = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.canvas_rect_id = None
        self.start_xy = None
        self.current_stage_idx = 0

        # SIMPLIFIED: `game_over` no longer needs a restart_area_roi.
        self.config_data = {
            "setup": {"ref_img_b64": None, "game_region": None},
            "gameplay": [],
            "menu": {"ref_img_b64": None, "start_button_roi": None},
            "game_over": {"ref_img_b64": None, "distance_meter_roi": None},
        }
        self.current_gameplay_set = {}

        self._build_ui()
        self._bind_hotkeys()
        self._update_ui_for_stage()

    def _build_ui(self):
        top = tk.Frame(self, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self.stage_label = tk.Label(top, text="", font=("Segoe UI", 12, "bold"))
        self.stage_label.pack()

        # --- Gameplay Tools (remains the same) ---
        self.gameplay_tool_row = tk.Frame(self, pady=6)
        tk.Label(
            self.gameplay_tool_row, text="Gameplay tool:", font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(10, 4))
        self.tool_var_gameplay = tk.StringVar(value=self.GAMEPLAY_TOOLS[0])
        for tool, key in zip(self.GAMEPLAY_TOOLS, ["F", "G", "B", "D"]):
            b = tk.Radiobutton(
                self.gameplay_tool_row,
                text=f"{tool.upper()} ({key})",
                value=tool,
                variable=self.tool_var_gameplay,
                command=self._on_tool_change,
            )
            b.pack(side=tk.LEFT, padx=6)

        # REMOVED: The game over tool row is no longer needed.

        self.canvas = tk.Canvas(self, bg="#2c2c2c")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas.bind("<Configure>", lambda e: self._render_image())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        bottom = tk.Frame(self, pady=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.instructions_label = tk.Label(
            bottom, text="", font=("Segoe UI", 11), justify=tk.LEFT
        )
        self.instructions_label.pack(side=tk.LEFT, padx=12)

        btns = tk.Frame(bottom)
        btns.pack(side=tk.RIGHT, padx=10)

        self.load_btn = tk.Button(
            btns, text="Load Screenshot", command=self._load_screenshot
        )
        self.load_btn.pack(side=tk.LEFT, padx=8)
        self.capture_btn = tk.Button(
            btns,
            text="Capture Screen",
            command=self._capture_screen_with_monitor_picker,
        )
        self.capture_btn.pack(side=tk.LEFT, padx=8)
        self.undo_btn = tk.Button(
            btns, text="Clear Current ROI", command=self._clear_current_roi
        )
        self.undo_btn.pack(side=tk.LEFT, padx=8)
        self.add_gameplay_btn = tk.Button(
            btns, text="✅ Add Gameplay Set", command=self._save_current_gameplay_set
        )
        self.next_btn = tk.Button(
            btns, text="Confirm & Next ▶", state=tk.DISABLED, command=self._next_stage
        )
        self.next_btn.pack(side=tk.LEFT, padx=8)
        self.save_btn = tk.Button(
            btns, text="Save Config JSON", state=tk.DISABLED, command=self._save_json
        )
        self.save_btn.pack(side=tk.LEFT, padx=8)

    def _bind_hotkeys(self):
        self.bind("<f>", lambda e: self._set_tool("fuel"))
        self.bind("<g>", lambda e: self._set_tool("gas"))
        self.bind("<b>", lambda e: self._set_tool("brake"))
        self.bind("<d>", lambda e: self._set_tool("distance"))  # ← add this
        self.bind("<Return>", lambda e: self._try_next())

    @property
    def current_stage(self):
        return self.STAGES[self.current_stage_idx]

    def _update_ui_for_stage(self):
        stage = self.current_stage

        self.gameplay_tool_row.pack_forget()
        self.add_gameplay_btn.pack_forget()
        self.next_btn.pack(side=tk.LEFT, padx=8)

        if stage == "GAMEPLAY":
            self.gameplay_tool_row.pack(side=tk.TOP, fill=tk.X)
            self.add_gameplay_btn.pack(side=tk.LEFT, padx=8)
            self.stage_label.config(
                text=f"Stage: {stage} (Sets Saved: {len(self.config_data['gameplay'])})"
            )
        else:  # Covers SETUP, MENU, and GAME_OVER
            self.stage_label.config(text=f"Stage: {stage}")

        # SIMPLIFIED: Instructions for the final step are clearer now.
        stage_texts = {
            "SETUP": "STEP 1 — SETUP:\n1. Get the game into its main gameplay view.\n2. Click 'Load Screenshot' or 'Capture Screen'.\n3. Draw the main GAME REGION rectangle.",
            "GAMEPLAY": "STEP 2 — GAMEPLAY (Collection):\n1. Load/Capture a gameplay screen.\n2. Draw ROIs for FUEL Bar, GAS Peddle, and BRAKE Peddle.\n3. Click 'Add Gameplay Set' to save. Repeat for more images.\n4. Click 'Confirm & Next' when done collecting.",
            "MENU": "STEP 3 — RESTART GAME:\n1. Navigate to the pages menu.\n2. 'Load' or 'Capture', then draw the START BUTTON area.",
            "GAME_OVER": "STEP 4 — DRIVER DOWN:\n1. Get to the game-over screen.\n2. 'Load' or 'Capture', then draw ONE box around the final DISTANCE meter.",
        }
        self.instructions_label.config(text=stage_texts.get(stage, ""))
        self._update_button_states()

    def _update_button_states(self):
        s = self.current_stage
        c = self.config_data

        self.next_btn.config(state=tk.DISABLED)
        self.add_gameplay_btn.config(state=tk.DISABLED)

        if s == "SETUP":
            if c["setup"].get("game_region"):
                self.next_btn.config(state=tk.NORMAL)
        elif s == "GAMEPLAY":
            if all(
                k in self.current_gameplay_set
                for k in [
                    "ref_img_b64",
                    "fuel_roi",
                    "gas_roi",
                    "brake_roi",
                    "distance_roi",
                ]
            ):
                self.add_gameplay_btn.config(state=tk.NORMAL)
            if len(c["gameplay"]) > 0:
                self.next_btn.config(state=tk.NORMAL)
        elif s == "MENU":
            if c["menu"].get("start_button_roi"):
                self.next_btn.config(state=tk.NORMAL)
        # SIMPLIFIED: Only check for the distance meter now.
        elif s == "GAME_OVER":
            if c["game_over"].get("distance_meter_roi"):
                self.next_btn.config(state=tk.NORMAL)

        # SIMPLIFIED: Final check no longer needs restart_area_roi.
        all_done = (
            c["setup"]["game_region"]
            and len(c["gameplay"]) > 0
            and c["menu"]["start_button_roi"]
            and c["game_over"]["distance_meter_roi"]
        )
        self.save_btn.config(state=tk.NORMAL if all_done else tk.DISABLED)

    def _process_new_image(self, pil_image):
        stage = self.current_stage
        if stage == "SETUP":
            self.current_img = pil_image
            self.config_data["setup"]["ref_img_b64"] = self._pil_to_b64(
                self.current_img
            )
        else:
            game_region = self.config_data["setup"]["game_region"]
            if not game_region:
                messagebox.showerror(
                    "Error",
                    "Game region not set. Please complete the SETUP stage first.",
                )
                return
            l, t, w, h = (
                game_region["left"],
                game_region["top"],
                game_region["width"],
                game_region["height"],
            )
            if pil_image.width < (l + w) or pil_image.height < (t + h):
                messagebox.showwarning(
                    "Warning",
                    "The loaded image is smaller than the defined game region.",
                )

            cropped_img = pil_image.crop((l, t, l + w, t + h)).copy()
            self.current_img = cropped_img
            b64_img = self._pil_to_b64(self.current_img)
            stage_key = stage.lower()

            if stage_key == "gameplay":
                self.current_gameplay_set = {"ref_img_b64": b64_img}
            elif stage_key in self.config_data:
                self.config_data[stage_key]["ref_img_b64"] = b64_img

        self._render_image()
        self._update_button_states()

    def _draw_existing_roi_overlay(self):
        s = self.current_stage
        c = self.config_data
        s_key = s.lower()

        if s == "SETUP" and c["setup"]["game_region"]:
            roi = c["setup"]["game_region"]
            self._draw_rect(
                roi["left"],
                roi["top"],
                roi["left"] + roi["width"],
                roi["top"] + roi["height"],
                "#00FFFF",
                "GAME REGION",
            )
        elif s == "GAMEPLAY":
            for name, color, label in [
                ("fuel_roi", "#00FF00", "FUEL"),
                ("gas_roi", "#FFD700", "GAS"),
                ("brake_roi", "#FF4500", "BRAKE"),
                ("distance_roi", "#00BFFF", "DISTANCE"),
            ]:
                if name in self.current_gameplay_set:
                    r = self.current_gameplay_set[name]
                    self._draw_rect(r["x1"], r["y1"], r["x2"], r["y2"], color, label)
        elif s_key == "menu" and c[s_key].get("start_button_roi"):
            r = c[s_key]["start_button_roi"]
            self._draw_rect(r["x1"], r["y1"], r["x2"], r["y2"], "#87CEEB", "START")
        # SIMPLIFIED: Only draw the distance ROI on the game over screen.
        elif s_key == "game_over":
            if c[s_key].get("distance_meter_roi"):
                r = c[s_key]["distance_meter_roi"]
                self._draw_rect(
                    r["x1"], r["y1"], r["x2"], r["y2"], "#9370DB", "DISTANCE"
                )

    def _on_release(self, event):
        if not self.start_xy or self.current_img is None:
            return
        x1c, y1c, x2c, y2c = *self.start_xy, event.x, event.y
        x1c, x2c = sorted((x1c, x2c))
        y1c, y2c = sorted((y1c, y2c))
        x1, y1 = self._canvas_to_image((x1c, y1c))
        x2, y2 = self._canvas_to_image((x2c, y2c))
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self._clear_temp_rect()
            self.start_xy = None
            return

        roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        s = self.current_stage

        if s == "SETUP":
            self.config_data["setup"]["game_region"] = {
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        elif s == "GAMEPLAY":
            self.current_gameplay_set[f"{self.tool_var_gameplay.get()}_roi"] = roi
        elif s == "MENU":
            self.config_data["menu"]["start_button_roi"] = roi
        # SIMPLIFIED: Only one possible annotation for game over.
        elif s == "GAME_OVER":
            self.config_data["game_over"]["distance_meter_roi"] = roi

        self._clear_temp_rect()
        self.start_xy = None
        self._render_image()
        self._update_button_states()

    def _clear_current_roi(self):
        s = self.current_stage
        if s == "SETUP":
            self.config_data["setup"]["game_region"] = None
        elif s == "GAMEPLAY":
            self.current_gameplay_set.pop(f"{self.tool_var_gameplay.get()}_roi", None)
        elif s == "MENU":
            self.config_data["menu"]["start_button_roi"] = None
        # SIMPLIFIED: Only one ROI to clear.
        elif s == "GAME_OVER":
            self.config_data["game_over"]["distance_meter_roi"] = None
        self._render_image()
        self._update_button_states()

    def _set_tool(self, tool):
        if self.current_stage == "GAMEPLAY" and tool in self.GAMEPLAY_TOOLS:
            self.tool_var_gameplay.set(tool)
        self._on_tool_change()

    def _on_tool_change(self):
        self._render_image()

    # --- Methods below are unchanged ---

    def _capture_screen_with_monitor_picker(self):
        selector = MonitorSelector(self)
        self.wait_window(selector)
        if not selector.selected_monitor:
            return
        time.sleep(0.2)
        with mss() as sct:
            sct_img = sct.grab(selector.selected_monitor)
            pil_image = Image.frombytes(
                "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"
            )
        self._process_new_image(pil_image)

    def _load_screenshot(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ]
        )
        if not filepath:
            return
        try:
            pil_image = Image.open(filepath).convert("RGB")
            self._process_new_image(pil_image)
        except Exception as e:
            messagebox.showerror(
                "Error Loading Image", f"Could not load the file:\n{e}"
            )

    def _render_image(self):
        self.canvas.delete("all")
        if self.current_img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        iw, ih = self.current_img.size
        self.scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        self.offset = ((cw - dw) // 2, (ch - dh) // 2)
        resized = self.current_img.resize((dw, dh), Image.Resampling.LANCZOS)
        self.current_tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset, anchor="nw", image=self.current_tk_img)
        self._draw_existing_roi_overlay()

    def _draw_rect(self, x1, y1, x2, y2, color, label):
        sx1, sy1 = self._image_to_canvas(x1, y1)
        sx2, sy2 = self._image_to_canvas(x2, y2)
        self.canvas.create_rectangle(
            sx1, sy1, sx2, sy2, outline=color, width=2, dash=(4, 4)
        )
        self.canvas.create_text(
            sx1 + 5,
            sy1 + 5,
            text=label,
            fill=color,
            anchor="nw",
            font=("Segoe UI", 10, "bold"),
        )

    def _save_current_gameplay_set(self):
        if self.current_stage != "GAMEPLAY":
            return
        self.config_data["gameplay"].append(self.current_gameplay_set.copy())
        self.current_gameplay_set = {}
        self.current_img = None
        self.canvas.delete("all")
        messagebox.showinfo(
            "Set Saved",
            f"Gameplay set saved! You now have {len(self.config_data['gameplay'])} sets.\nLoad another image to continue or click 'Next'.",
        )
        self._update_ui_for_stage()

    def _next_stage(self):
        if self.current_stage_idx < len(self.STAGES) - 1:
            self.current_stage_idx += 1
            self.current_img = None
            self.canvas.delete("all")
            self._update_ui_for_stage()

    def _try_next(self):
        if self.next_btn["state"] == tk.NORMAL:
            self._next_stage()
        elif (
            self.add_gameplay_btn.winfo_ismapped()
            and self.add_gameplay_btn["state"] == tk.NORMAL
        ):
            self._save_current_gameplay_set()

    def _save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="hcr_config.json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self.config_data, f, indent=2)
        messagebox.showinfo(
            "Saved", f"Configuration saved to:\n{os.path.abspath(path)}"
        )

    def _on_press(self, event):
        if self.current_img is None:
            return
        self.start_xy = (event.x, event.y)
        self._clear_temp_rect()
        self.canvas_rect_id = self.canvas.create_rectangle(
            self.start_xy, self.start_xy, outline="#FFFFFF", width=2, dash=(5, 3)
        )

    def _on_drag(self, event):
        if self.start_xy:
            self.canvas.coords(self.canvas_rect_id, *self.start_xy, event.x, event.y)

    def _clear_temp_rect(self):
        if self.canvas_rect_id:
            self.canvas.delete(self.canvas_rect_id)
        self.canvas_rect_id = None

    def _pil_to_b64(self, pil_image):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _canvas_to_image(self, c_coords):
        if not self.current_img or self.scale == 0:
            return 0, 0
        x = int(
            max(
                0,
                min(
                    self.current_img.size[0],
                    (c_coords[0] - self.offset[0]) / self.scale,
                ),
            )
        )
        y = int(
            max(
                0,
                min(
                    self.current_img.size[1],
                    (c_coords[1] - self.offset[1]) / self.scale,
                ),
            )
        )
        return x, y

    def _image_to_canvas(self, x, y):
        return (self.offset[0] + x * self.scale, self.offset[1] + y * self.scale)


if __name__ == "__main__":
    app = MultiScreenAnnotator()
    app.mainloop()
