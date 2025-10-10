import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import base64, io, os, time
from mss import mss

# -------------------------------
# Monitor picker (unchanged idea)
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

        tk.Label(self, text="Click the monitor you want to capture.", font=("Segoe UI", 14)).pack(pady=10)
        self.monitor_frame = tk.Frame(self)
        self.monitor_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.populate_monitors()

    def populate_monitors(self):
        with mss() as sct:
            for i, monitor in enumerate(sct.monitors[1:]):
                frame = tk.Frame(self.monitor_frame, borderwidth=2, relief="groove")
                frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=5)

                img_sct = sct.grab(monitor)
                img_pil = Image.frombytes("RGB", img_sct.size, img_sct.bgra, "raw", "BGRX")
                img_pil.thumbnail((320, 180))
                img_tk = ImageTk.PhotoImage(img_pil)
                self.preview_images.append(img_tk)

                tk.Button(frame, image=img_tk, command=lambda m=monitor: self.select_monitor(m)).pack(padx=5, pady=5)
                tk.Label(frame, text=f"Monitor {i+1}", font=("Segoe UI", 10)).pack(pady=5)

    def select_monitor(self, monitor_details):
        self.selected_monitor = monitor_details
        self.destroy()

# -------------------------------
# Main Annotator
# -------------------------------
class MultiScreenAnnotator(tk.Tk):
    STAGES = ["SETUP", "GAMEPLAY", "MENU", "GAME_OVER"]
    GAMEPLAY_TOOLS = ["fuel", "gas", "brake"]  # three ROIs you asked for

    def __init__(self):
        super().__init__()
        self.title("HCR - Final Annotator")
        self.geometry("1200x800")
        self.minsize(900, 600)

        # rendering state
        self.current_img = None        # full captured (or cropped-to-game-region) PIL image
        self.current_tk_img = None
        self.scale = 1.0
        self.offset = (0, 0)

        # draw state
        self.canvas_rect_id = None
        self.start_xy = None

        # stage/tool state
        self.current_stage_idx = 0
        self.active_tool = "fuel"  # default in GAMEPLAY

        # config result model
        self.config_data = {
            "setup": {"ref_img_b64": None, "game_region": None},  # SETUP stores reference and region
            "gameplay": {
                "ref_img_b64": None,
                "fuel_roi": None,
                "gas_roi": None,
                "brake_roi": None,
            },
            "menu": {"ref_img_b64": None, "start_button_roi": None},
            "game_over": {"ref_img_b64": None, "restart_area_roi": None},
        }

        # UI
        self._build_ui()
        self._bind_hotkeys()
        self._update_instructions()

    # ---------- UI ----------
    def _build_ui(self):
        # Header
        top = tk.Frame(self, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self.stage_label = tk.Label(top, text="", font=("Segoe UI", 12, "bold"))
        self.stage_label.pack()

        # Gameplay tool row (hidden unless stage == GAMEPLAY)
        tool_row = tk.Frame(self, pady=6)
        tool_row.pack(side=tk.TOP, fill=tk.X)
        self.tool_row = tool_row

        tk.Label(tool_row, text="Gameplay tool:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(10,4))
        self.tool_var = tk.StringVar(value=self.active_tool)
        for tool, key in zip(self.GAMEPLAY_TOOLS, ["F", "G", "B"]):
            b = tk.Radiobutton(tool_row, text=f"{tool.upper()} ({key})", value=tool,
                               variable=self.tool_var, command=self._on_tool_change)
            b.pack(side=tk.LEFT, padx=6)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#2c2c2c")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas.bind("<Configure>", lambda e: self._render_image())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Footer
        bottom = tk.Frame(self, pady=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.instructions_label = tk.Label(bottom, text="", font=("Segoe UI", 11))
        self.instructions_label.pack(side=tk.LEFT, padx=12)

        btns = tk.Frame(bottom)
        btns.pack(side=tk.RIGHT)

        self.capture_btn = tk.Button(btns, text="Capture Screen", command=self._capture_screen_with_monitor_picker)
        self.capture_btn.pack(side=tk.LEFT, padx=8)

        self.undo_btn = tk.Button(btns, text="Clear Current ROI", command=self._clear_current_roi)
        self.undo_btn.pack(side=tk.LEFT, padx=8)

        self.next_btn = tk.Button(btns, text="Confirm & Next ▶", state=tk.DISABLED, command=self._next_stage)
        self.next_btn.pack(side=tk.LEFT, padx=8)

        self.save_btn = tk.Button(btns, text="Save Config JSON", state=tk.DISABLED, command=self._save_json)
        self.save_btn.pack(side=tk.LEFT, padx=8)

    def _bind_hotkeys(self):
        # gameplay tool shortcuts
        self.bind("<f>", lambda e: self._set_tool("fuel"))
        self.bind("<g>", lambda e: self._set_tool("gas"))
        self.bind("<b>", lambda e: self._set_tool("brake"))
        # navigation helpers
        self.bind("<Return>", lambda e: self._try_next())

    # ---------- Stage / UI helpers ----------
    @property
    def current_stage(self):
        return self.STAGES[self.current_stage_idx]

    def _update_instructions(self):
        self.stage_label.config(text=f"Stage: {self.current_stage}")
        # show tool row only for GAMEPLAY
        if self.current_stage == "GAMEPLAY":
            self.tool_row.pack_configure(side=tk.TOP, fill=tk.X)
        else:
            self.tool_row.pack_forget()

        if self.current_stage == "SETUP":
            txt = "STEP 1 — SETUP:\n1) Put game in GAMEPLAY view.\n2) Click 'Capture Screen'.\n3) Draw the GAME REGION rectangle."
        elif self.current_stage == "GAMEPLAY":
            txt = "STEP 2 — GAMEPLAY:\nClick 'Capture Screen' (region-cropped).\nDraw ROIs for FUEL (F), GAS (G), BRAKE (B)."
        elif self.current_stage == "MENU":
            txt = "STEP 3 — MENU:\nOpen the menu, 'Capture Screen', then draw START BUTTON area."
        else:  # GAME_OVER
            txt = "STEP 4 — GAME OVER:\nOpen game-over screen, 'Capture Screen', then draw RESTART area."
        self.instructions_label.config(text=txt)
        self._update_next_enabled()

    def _update_next_enabled(self):
        ok = False
        s = self.current_stage
        if s == "SETUP":
            ok = self.config_data["setup"]["game_region"] is not None and self.config_data["setup"]["ref_img_b64"] is not None
        elif s == "GAMEPLAY":
            g = self.config_data["gameplay"]
            ok = all([g["ref_img_b64"], g["fuel_roi"], g["gas_roi"], g["brake_roi"]])
        elif s == "MENU":
            m = self.config_data["menu"]
            ok = m["ref_img_b64"] is not None and m["start_button_roi"] is not None
        elif s == "GAME_OVER":
            go = self.config_data["game_over"]
            ok = go["ref_img_b64"] is not None and go["restart_area_roi"] is not None

        self.next_btn.config(state=(tk.NORMAL if ok else tk.DISABLED))
        # Allow saving only after last stage complete
        all_done = (
            self.config_data["setup"]["game_region"] and
            self.config_data["gameplay"]["fuel_roi"] and
            self.config_data["gameplay"]["gas_roi"] and
            self.config_data["gameplay"]["brake_roi"] and
            self.config_data["menu"]["start_button_roi"] and
            self.config_data["game_over"]["restart_area_roi"]
        )
        self.save_btn.config(state=(tk.NORMAL if all_done else tk.DISABLED))

    # ---------- Capture / Images ----------
    def _capture_screen_with_monitor_picker(self):
        selector = MonitorSelector(self)
        self.wait_window(selector)
        if not selector.selected_monitor:
            return

        time.sleep(0.2)
        with mss() as sct:
            sct_img = sct.grab(selector.selected_monitor)
            raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # On SETUP, we display full monitor; after region is set we crop to region on each capture
        if self.current_stage == "SETUP":
            self.current_img = raw
            # store a reference image for setup too (optional but nice)
            self.config_data["setup"]["ref_img_b64"] = self._pil_to_b64(self.current_img)
            self._render_image()
        else:
            region = self.config_data["setup"]["game_region"]
            if not region:
                messagebox.showerror("Error", "Game region not set. Complete SETUP first.")
                return
            l, t = region["left"], region["top"]
            w, h = region["width"], region["height"]
            cropped = raw.crop((l, t, l + w, t + h)).copy()
            self.current_img = cropped
            # store stage reference
            if self.current_stage == "GAMEPLAY":
                self.config_data["gameplay"]["ref_img_b64"] = self._pil_to_b64(self.current_img)
            elif self.current_stage == "MENU":
                self.config_data["menu"]["ref_img_b64"] = self._pil_to_b64(self.current_img)
            elif self.current_stage == "GAME_OVER":
                self.config_data["game_over"]["ref_img_b64"] = self._pil_to_b64(self.current_img)
            self._render_image()

        self._update_next_enabled()

    def _render_image(self):
        self.canvas.delete("all")
        if self.current_img is None:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        self.scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        self.offset = ((cw - dw) // 2, (ch - dh) // 2)

        resized = self.current_img.resize((dw, dh), Image.Resampling.LANCZOS)
        self.current_tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset, anchor="nw", image=self.current_tk_img)

        # Repaint existing rectangle (if any)
        self._draw_existing_roi_overlay()

    def _draw_existing_roi_overlay(self):
        s = self.current_stage
        if s == "SETUP":
            roi = self.config_data["setup"]["game_region"]
            if roi:
                self._draw_rect(roi["left"], roi["top"], roi["left"] + roi["width"], roi["top"] + roi["height"], color="#00FFFF")
        elif s == "GAMEPLAY":
            g = self.config_data["gameplay"]
            for name, color in [("fuel_roi", "#00FF00"), ("gas_roi", "#FFD700"), ("brake_roi", "#FF4500")]:
                r = g.get(name)
                if r:
                    self._draw_rect(r["x1"], r["y1"], r["x2"], r["y2"], color=color)
        elif s == "MENU":
            r = self.config_data["menu"]["start_button_roi"]
            if r:
                self._draw_rect(r["x1"], r["y1"], r["x2"], r["y2"], color="#87CEEB")
        elif s == "GAME_OVER":
            r = self.config_data["game_over"]["restart_area_roi"]
            if r:
                self._draw_rect(r["x1"], r["y1"], r["x2"], r["y2"], color="#FF69B4")

    def _draw_rect(self, x1, y1, x2, y2, color="#00FF00"):
        sx1, sy1 = self._image_to_canvas(x1, y1)
        sx2, sy2 = self._image_to_canvas(x2, y2)
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2, dash=(4, 4))

    # ---------- Mouse / ROI ----------
    def _on_press(self, event):
        if self.current_img is None:
            return
        self.start_xy = (event.x, event.y)
        if self.canvas_rect_id:
            self.canvas.delete(self.canvas_rect_id)
            self.canvas_rect_id = None
        self.canvas_rect_id = self.canvas.create_rectangle(self.start_xy, self.start_xy,
                                                           outline="#00FF00", width=2, dash=(4, 4))

    def _on_drag(self, event):
        if self.start_xy:
            self.canvas.coords(self.canvas_rect_id, self.start_xy[0], self.start_xy[1], event.x, event.y)

    def _on_release(self, event):
        if not self.start_xy or self.current_img is None:
            return
        x1c, y1c = self.start_xy
        x2c, y2c = event.x, event.y

        # normalize + convert to image coords
        x1c, x2c = sorted((x1c, x2c))
        y1c, y2c = sorted((y1c, y2c))
        x1, y1 = self._canvas_to_image((x1c, y1c))
        x2, y2 = self._canvas_to_image((x2c, y2c))
        x1, x2 = int(max(0, x1)), int(max(0, x2))
        y1, y2 = int(max(0, y1)), int(max(0, y2))

        # discard tiny rectangles
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            self._clear_temp_rect()
            self.start_xy = None
            return

        s = self.current_stage
        if s == "SETUP":
            # store game region
            roi = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
            self.config_data["setup"]["game_region"] = roi
            # immediately crop and set gameplay ref image
            l, t, w, h = roi["left"], roi["top"], roi["width"], roi["height"]
            self.current_img = self.current_img.crop((l, t, l + w, t + h)).copy()
            self.config_data["gameplay"]["ref_img_b64"] = self._pil_to_b64(self.current_img)

        elif s == "GAMEPLAY":
            roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            tool = self.tool_var.get()
            key = f"{tool}_roi"  # fuel_roi, gas_roi, brake_roi
            self.config_data["gameplay"][key] = roi

        elif s == "MENU":
            self.config_data["menu"]["start_button_roi"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        elif s == "GAME_OVER":
            self.config_data["game_over"]["restart_area_roi"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        self._clear_temp_rect()
        self.start_xy = None
        self._render_image()
        self._update_next_enabled()

    def _clear_temp_rect(self):
        if self.canvas_rect_id:
            self.canvas.delete(self.canvas_rect_id)
            self.canvas_rect_id = None

    def _clear_current_roi(self):
        s = self.current_stage
        if s == "SETUP":
            self.config_data["setup"]["game_region"] = None
        elif s == "GAMEPLAY":
            key = f"{self.tool_var.get()}_roi"
            self.config_data["gameplay"][key] = None
        elif s == "MENU":
            self.config_data["menu"]["start_button_roi"] = None
        elif s == "GAME_OVER":
            self.config_data["game_over"]["restart_area_roi"] = None
        self._render_image()
        self._update_next_enabled()

    # ---------- Navigation ----------
    def _next_stage(self):
        # Freeze reference image (already saved) and advance
        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.STAGES):
            self.current_stage_idx = len(self.STAGES) - 1  # stay at last
        self.current_img = None
        self._update_instructions()
        self.canvas.delete("all")

    def _try_next(self):
        if self.next_btn["state"] == tk.NORMAL:
            self._next_stage()

    # ---------- Utils ----------
    def _on_tool_change(self):
        self.active_tool = self.tool_var.get()
        self._render_image()

    def _set_tool(self, tool):
        if self.current_stage != "GAMEPLAY":
            return
        self.tool_var.set(tool)
        self._on_tool_change()

    def _save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="hcr_config.json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return

        import json
        with open(path, "w") as f:
            json.dump(self.config_data, f, indent=2)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{os.path.abspath(path)}")

    def _pil_to_b64(self, pil_image):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _canvas_to_image(self, canvas_coords):
        x = (canvas_coords[0] - self.offset[0]) / self.scale
        y = (canvas_coords[1] - self.offset[1]) / self.scale
        # clamp to image bounds
        x = max(0, min(self.current_img.size[0], x))
        y = max(0, min(self.current_img.size[1], y))
        return x, y

    def _image_to_canvas(self, x, y):
        return (self.offset[0] + x * self.scale, self.offset[1] + y * self.scale)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app = MultiScreenAnnotator()
    app.mainloop()
