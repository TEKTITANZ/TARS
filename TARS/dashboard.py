import customtkinter as ctk
import cv2
import math
import time
import threading
import queue
import os
from datetime import datetime
from PIL import Image
from tracker import AstrotrackEngine, experiment_steps

# --- Strict 1:1 Color Palette ---
BG_MAIN = "#121419"        
BG_PANEL = "#1C2028"       
BG_SUBPANEL = "#252B36"    
TEXT_LIGHT = "#E2E8F0"
TEXT_MUTED = "#94A3B8"
ACCENT_BLUE = "#00AAFF"    
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"

# ==========================================
# ASYNCHRONOUS VISION THREAD
# ==========================================
class VisionThread(threading.Thread):
    """Capture + AI processing thread.

    The GUI never touches the camera or YOLO model directly. A single paired
    result queue prevents a frame and HUD message from getting out of sync.
    """
    def __init__(self, camera_index=1):
        super().__init__(daemon=True)

        # WEBCAM ONLY: no desktop/screen capture is used anywhere in this app.
        # Try the requested USB webcam index first, then common Windows camera
        # indexes. The first camera that can actually return a frame is used.
        self.camera_index = camera_index
        self.engine = AstrotrackEngine(camera_index=camera_index)

        self.cap, self.camera_index = self._find_webcam(camera_index)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "No webcam could be opened. Connect the USB webcam and "
                "close other applications using the camera, then restart AstroTrack."
            )

        # Match the reference-video style while keeping inference responsive.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.result_queue = queue.Queue(maxsize=2)
        self.running = True

    @staticmethod
    def _open_camera(index):
        """Open a physical webcam using Windows DirectShow, then MSMF."""
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Verify that this is a real usable camera, not just an index.
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        return cap
                cap.release()
            else:
                cap.release()
        return None

    @classmethod
    def _find_webcam(cls, preferred_index):
        """Find a working physical webcam. No screen/window capture is attempted."""
        candidates = [preferred_index, 1, 0, 2, 3, 4, 5]
        seen = set()
        for index in candidates:
            if index in seen:
                continue
            seen.add(index)
            cap = cls._open_camera(index)
            if cap is not None:
                print(f"[Camera] Using physical webcam index {index}")
                return cap, index
        return None, preferred_index

    def run(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.03)
                continue

            try:
                annotated, hud = self.engine.process_frame(frame)
            except Exception as exc:
                # Keep the GUI alive if one bad frame/model call occurs.
                print(f"[VisionThread] frame processing error: {exc}")
                time.sleep(0.03)
                continue

            # Drop the oldest complete result, never a frame without its HUD.
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.result_queue.put_nowait((annotated, hud))
            except queue.Full:
                pass

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.engine.close()
        except Exception:
            pass


class AstrotrackDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ISRO-Astrotrack: Edge Operator Console")
        self.state("zoomed") 
        self.configure(fg_color=BG_MAIN)
        
        self.start_time = time.time()
        
        # --- Root Grid ---
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # ==========================================
        # TOP HEADER: MET TIMER
        # ==========================================
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 0))
        
        self.met_label = ctk.CTkLabel(
            self.header_frame, text="🔴 MET: 00:00:00", font=("Arial", 22), text_color=TEXT_LIGHT
        )
        self.met_label.pack(side="left")

        # ==========================================
        # MAIN CONTENT GRID
        # ==========================================
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=1, column=0, sticky="nsew", padx=25, pady=20)
        self.main_content.grid_columnconfigure(0, weight=45) 
        self.main_content.grid_columnconfigure(1, weight=25) 
        self.main_content.grid_columnconfigure(2, weight=30) 
        self.main_content.grid_rowconfigure(0, weight=1)

        # ------------------------------------------
        # COLUMN 1: VIDEO & STATUS
        # ------------------------------------------
        self.col1 = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.col1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_panel = ctk.CTkFrame(self.col1, fg_color=BG_PANEL, corner_radius=12)
        self.video_panel.pack(fill="both", expand=True, pady=(0, 10))
        
        self.badge_frame = ctk.CTkFrame(self.video_panel, fg_color="#333333", corner_radius=4)
        self.badge_frame.pack(anchor="nw", padx=15, pady=(15, 0))
        ctk.CTkLabel(self.badge_frame, text=" LIVE HD WEBCAM ", font=("Arial", 11, "bold"), text_color="#FFFFFF").pack(padx=5, pady=2)
        
        self.video_label = ctk.CTkLabel(self.video_panel, text="")
        self.video_label.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        
        self.hw_status = ctk.CTkFrame(self.video_panel, fg_color=BG_SUBPANEL, corner_radius=6, height=45)
        self.hw_status.pack(fill="x", padx=15, pady=(0, 15))
        self.hw_status.pack_propagate(False)
        ctk.CTkLabel(self.hw_status, text="◉ WEBCAM: ACTIVE", font=("Arial", 12, "bold"), text_color=ACCENT_GREEN).pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(self.hw_status, text="🎤 MIC: ON", font=("Arial", 12, "bold"), text_color=ACCENT_GREEN).pack(side="left", padx=20, pady=12)

        self.ai_panel_left = ctk.CTkFrame(self.col1, fg_color=BG_PANEL, corner_radius=12, height=160)
        self.ai_panel_left.pack(fill="x")
        self.ai_panel_left.pack_propagate(False)
        self.wave_canvas_left = ctk.CTkCanvas(self.ai_panel_left, bg=BG_PANEL, highlightthickness=0)
        self.wave_canvas_left.pack(fill="both", expand=True, padx=10, pady=10)

        # ------------------------------------------
        # COLUMN 2: INSTRUCTIONS & PROGRESS
        # ------------------------------------------
        self.col2 = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.col2.grid(row=0, column=1, sticky="nsew", padx=10)
        
        self.ai_panel_center = ctk.CTkFrame(self.col2, fg_color=BG_PANEL, corner_radius=12, height=140)
        self.ai_panel_center.pack(fill="x", pady=(0, 10))
        self.ai_panel_center.pack_propagate(False)
        self.wave_canvas_center = ctk.CTkCanvas(self.ai_panel_center, bg=BG_PANEL, highlightthickness=0)
        self.wave_canvas_center.pack(fill="both", expand=True, padx=10, pady=10)

        self.inst_panel = ctk.CTkFrame(self.col2, fg_color=BG_PANEL, corner_radius=12)
        self.inst_panel.pack(fill="both", expand=True)
        ctk.CTkLabel(self.inst_panel, text="ACTIVE INSTRUCTIONS", font=("Arial", 14, "bold"), text_color=TEXT_LIGHT).pack(anchor="nw", padx=20, pady=(20, 10))
        
        self.inst_list_frame = ctk.CTkFrame(self.inst_panel, fg_color="transparent")
        self.inst_list_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        self.step_ui_elements = []
        for i, step in enumerate(experiment_steps):
            step_container = ctk.CTkFrame(self.inst_list_frame, fg_color="transparent")
            step_container.pack(fill="x", pady=8)
            title = ctk.CTkLabel(step_container, text=f"{i+1}. {step['action']}", font=("Arial", 13), text_color=TEXT_LIGHT, anchor="w")
            title.pack(fill="x")
            sub = ctk.CTkLabel(step_container, text="Status: Pending", font=("Arial", 11), text_color=TEXT_MUTED, anchor="w")
            sub.pack(fill="x", padx=15)
            self.step_ui_elements.append({"title": title, "sub": sub})

        self.prog_container = ctk.CTkFrame(self.inst_panel, fg_color="transparent")
        self.prog_container.pack(fill="x", side="bottom", padx=20, pady=20)
        self.prog_labels = ctk.CTkFrame(self.prog_container, fg_color="transparent")
        self.prog_labels.pack(fill="x")
        ctk.CTkLabel(self.prog_labels, text="EXPERIMENT PROGRESS:", font=("Arial", 11, "bold"), text_color=TEXT_LIGHT).pack(side="left")
        self.prog_pct = ctk.CTkLabel(self.prog_labels, text="0%", font=("Arial", 12, "bold"), text_color=TEXT_LIGHT)
        self.prog_pct.pack(side="right")
        self.progress_bar = ctk.CTkProgressBar(self.prog_container, progress_color=ACCENT_BLUE, fg_color=BG_SUBPANEL, height=10)
        self.progress_bar.pack(fill="x", pady=(8, 0))
        self.progress_bar.set(0.0)

        # ------------------------------------------
        # COLUMN 3: ACTIVITY LOG
        # ------------------------------------------
        self.col3 = ctk.CTkFrame(self.main_content, fg_color=BG_PANEL, corner_radius=12)
        self.col3.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        ctk.CTkLabel(self.col3, text="ACTIVITY LOG", font=("Arial", 14, "bold"), text_color=TEXT_LIGHT).pack(anchor="nw", padx=20, pady=(20, 10))
        self.timeline_frame = ctk.CTkScrollableFrame(self.col3, fg_color="transparent")
        self.timeline_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ==========================================
        # STATE VARIABLES & START ENGINE
        # ==========================================
        self.last_anomaly = None
        self.completed_history_length = 0
        self.is_speaking = False
        self.speak_timer = 0
        self.wave_radii = []
        self.wave_spawn_timer = 0
        
        self.add_timeline_entry("Session Started", is_system=True)
        self.update_clock()
        self.animate_wave()
        
        # Physical USB webcam only. The camera code automatically tests
        # common Windows webcam indexes if the preferred index is unavailable.
        camera_index = int(os.environ.get("ASTROTRACK_CAMERA", "1"))
        self.vision_thread = VisionThread(camera_index=camera_index)
        self.vision_thread.start()
        
        self.update_gui_loop()

    def update_clock(self):
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.met_label.configure(text=f"🔴 MET: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.after(1000, self.update_clock)

    def add_timeline_entry(self, message, is_anomaly=False, is_success=False, is_system=False):
        entry_frame = ctk.CTkFrame(self.timeline_frame, fg_color="transparent")
        entry_frame.pack(fill="x", pady=6)
        now = datetime.now().strftime("%H:%M:%S")
        
        if is_anomaly:
            icon, color = "🔴", ACCENT_RED
        elif is_success:
            icon, color = "✅", ACCENT_GREEN
        elif is_system:
            icon, color = "🔵", ACCENT_BLUE
        else:
            icon, color = "○", TEXT_MUTED
            
        icon_lbl = ctk.CTkLabel(entry_frame, text=icon, font=("Arial", 12), text_color=color, width=20)
        icon_lbl.pack(side="left", anchor="n")
        text_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", padx=5)
        ctk.CTkLabel(text_frame, text=now, font=("Arial", 11), text_color=TEXT_MUTED).pack(anchor="w")
        ctk.CTkLabel(text_frame, text=message, font=("Arial", 12), text_color=TEXT_LIGHT, justify="left").pack(anchor="w")
        self.timeline_frame._parent_canvas.yview_moveto(1.0)

    def animate_wave(self):
        if self.is_speaking:
            for i in range(len(self.wave_radii)):
                self.wave_radii[i] += 2.5 
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                self.wave_radii.append(35) 
                self.wave_spawn_timer = 10 
        else:
            self.wave_radii.clear()
            self.wave_spawn_timer = 0
            
        for canvas in (self.wave_canvas_left, self.wave_canvas_center):
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                continue
            cx, cy = w / 2, h / 2
            canvas.create_oval(cx-30, cy-30, cx+30, cy+30, fill="#0A3C6E", outline="")
            canvas.create_oval(cx-24, cy-24, cx+24, cy+24, fill="#1263B3", outline="")
            canvas.create_oval(cx-18, cy-18, cx+18, cy+18, fill="#FFFFFF", outline="")
            
            if self.is_speaking:
                self.wave_radii[:] = [r for r in self.wave_radii if r < (w / 2)]
                for r in self.wave_radii:
                    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=135, extent=90, style="arc", outline=ACCENT_BLUE, width=2, dash=(2, 6))
                    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=315, extent=90, style="arc", outline=ACCENT_BLUE, width=2, dash=(2, 6))
            else:
                for r in [45, 65, 85]:
                    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=135, extent=90, style="arc", outline="#2A3B4C", width=1, dash=(2, 6))
                    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=315, extent=90, style="arc", outline="#2A3B4C", width=1, dash=(2, 6))
                    
        if self.is_speaking:
            self.speak_timer -= 1
            if self.speak_timer <= 0:
                self.is_speaking = False
                
        self.after(25, self.animate_wave)

    def trigger_audio_visualizer(self, duration_frames=80):
        self.is_speaking = True
        self.speak_timer = duration_frames

    def update_gui_loop(self):
        # Pull one complete frame+HUD pair.
        try:
            annotated, hud = self.vision_thread.result_queue.get_nowait()
        except queue.Empty:
            annotated = None
            hud = None

        if annotated is not None and hud is not None:
            
            step_idx = hud.get("step_index", 0)
            is_done = hud.get("done", False)
            
            for i, ui in enumerate(self.step_ui_elements):
                if i < step_idx:
                    ui["title"].configure(text_color=TEXT_MUTED)
                    ui["sub"].configure(text="Status: Completed", text_color=ACCENT_GREEN)
                elif i == step_idx and not is_done:
                    ui["title"].configure(text_color=TEXT_LIGHT, font=("Arial", 13, "bold"))
                    now_str = datetime.now().strftime("%H:%M")
                    ui["sub"].configure(text=f"Time: {now_str}", text_color=ACCENT_BLUE)
                else:
                    ui["title"].configure(text_color=TEXT_MUTED, font=("Arial", 13, "normal"))
                    ui["sub"].configure(text="Status: Pending", text_color=TEXT_MUTED)

            progress_pct = 1.0 if is_done else step_idx / len(experiment_steps)
            self.progress_bar.set(progress_pct)
            self.prog_pct.configure(text=f"{int(progress_pct * 100)}%")
            
            completed_events = self.vision_thread.engine.state.completed_history()
            if len(completed_events) > self.completed_history_length:
                latest = completed_events[-1]
                self.add_timeline_entry(f"Verified:\n{latest['action']}", is_success=True)
                self.completed_history_length = len(completed_events)
                self.trigger_audio_visualizer() 
            
            current_anomaly = self.vision_thread.engine.state.last_anomaly()
            if current_anomaly and current_anomaly != self.last_anomaly:
                self.add_timeline_entry(f"AI Warning:\n{current_anomaly}", is_anomaly=True)
                self.last_anomaly = current_anomaly
                self.trigger_audio_visualizer() 
            
            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            target_width = self.video_panel.winfo_width() - 30
            target_height = self.video_panel.winfo_height() - 75 
            
            if target_width > 100 and target_height > 100:
                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_width, target_height))
                self.video_label.configure(image=ctk_image)
                self.video_label.image = ctk_image
            
        # Call this much C:\Users\prana\OneDrive\Desktop\clg\python\TARS\dashboard.pfaster (e.g. 10ms) because it only checks the queue, it doesn't do the heavy math
        self.after(10, self.update_gui_loop)
        
    def on_closing(self):
        self.add_timeline_entry("System Shutting Down...", is_system=True)
        self.vision_thread.stop()
        self.after(50, self.destroy)

if __name__ == "__main__":
    app = AstrotrackDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()