"""Tkinter operator GUI: live camera, 'learn fabric', continuous inspection."""
import base64
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2

from .engine import Inspector, center_square, list_profiles, read_images

VIEW = 512


def _photo(bgr):
    ok, buf = cv2.imencode(".png", cv2.resize(bgr, (VIEW, VIEW)))
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class App:
    def __init__(self, root):
        self.root = root
        root.title("FabricInspect - pre-seam fabric inspection")
        self.cap = None
        self.folder_frames, self.folder_i = [], 0
        self.frame = None
        self.ins = Inspector()
        self.inspecting = False
        self.busy = False
        self.collect = []        # frames gathered during learning
        self.collect_target = 0
        self.last_collect = 0.0

        self.canvas = tk.Label(root, bg="black", width=VIEW, height=VIEW)
        self.canvas.grid(row=0, column=0, rowspan=20, padx=6, pady=6)
        p = ttk.Frame(root); p.grid(row=0, column=1, sticky="n", padx=6, pady=6)

        ttk.Label(p, text="Source").grid(row=0, column=0, sticky="w")
        self.cam_idx = tk.StringVar(value="0")
        ttk.Entry(p, textvariable=self.cam_idx, width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(p, text="Open camera", command=self.open_camera).grid(row=0, column=2)
        ttk.Button(p, text="Image folder...", command=self.open_folder).grid(row=1, column=2)

        ttk.Separator(p).grid(row=2, columnspan=3, sticky="ew", pady=6)
        ttk.Label(p, text="Fabric name").grid(row=3, column=0, sticky="w")
        self.fabric = tk.StringVar(value="fabric_1")
        ttk.Entry(p, textvariable=self.fabric, width=18).grid(row=3, column=1, columnspan=2, sticky="w")
        ttk.Label(p, text="Detector").grid(row=4, column=0, sticky="w")
        self.kind = tk.StringVar(value="patchcore")
        ttk.Radiobutton(p, text="Deep (PatchCore-lite)", variable=self.kind,
                        value="patchcore").grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(p, text="Light (Gabor-Gaussian)", variable=self.kind,
                        value="gabor").grid(row=5, column=1, columnspan=2, sticky="w")
        ttk.Label(p, text="Good shots").grid(row=6, column=0, sticky="w")
        self.shots = tk.IntVar(value=10)
        ttk.Spinbox(p, from_=2, to=100, textvariable=self.shots, width=6).grid(row=6, column=1, sticky="w")
        ttk.Label(p, text="ROI (0-1)").grid(row=7, column=0, sticky="w")
        self.roi = tk.DoubleVar(value=1.0)
        ttk.Entry(p, textvariable=self.roi, width=6).grid(row=7, column=1, sticky="w")
        ttk.Button(p, text="Learn fabric (move defect-free fabric slowly)",
                   command=self.start_learning).grid(row=8, column=0, columnspan=3, sticky="ew", pady=4)

        ttk.Label(p, text="Profile").grid(row=9, column=0, sticky="w")
        self.profile = ttk.Combobox(p, values=list_profiles(), width=16)
        self.profile.grid(row=9, column=1)
        ttk.Button(p, text="Load", command=self.load_profile).grid(row=9, column=2)

        ttk.Label(p, text="Webhook URL (optional)").grid(row=10, column=0, columnspan=3, sticky="w")
        self.webhook = tk.StringVar(value="")
        ttk.Entry(p, textvariable=self.webhook, width=34).grid(row=11, column=0, columnspan=3, sticky="w")

        self.btn_inspect = ttk.Button(p, text="Start inspection", command=self.toggle_inspect)
        self.btn_inspect.grid(row=12, column=0, columnspan=3, sticky="ew", pady=8)

        self.verdict = tk.Label(p, text="-", font=("Segoe UI", 28, "bold"), width=10)
        self.verdict.grid(row=13, column=0, columnspan=3, pady=4)
        self.info = ttk.Label(p, text="", justify="left")
        self.info.grid(row=14, column=0, columnspan=3, sticky="w")
        self.status = ttk.Label(p, text="Open a camera or an image folder.", wraplength=300,
                                foreground="#444")
        self.status.grid(row=15, column=0, columnspan=3, sticky="w", pady=6)
        self.tick()

    # ---- sources ------------------------------------------------------------
    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.folder_frames = []
        self.cap = cv2.VideoCapture(int(self.cam_idx.get()), cv2.CAP_DSHOW)
        ok = self.cap.isOpened()
        self.say("Camera opened." if ok else "Cannot open camera " + self.cam_idx.get())

    def open_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        if self.cap:
            self.cap.release(); self.cap = None
        self.folder_frames = read_images(d)
        self.folder_i = 0
        self.say(f"{len(self.folder_frames)} images loaded from folder.")

    def next_frame(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            return f if ok else None
        if self.folder_frames:
            # advance one image per second so the operator can follow
            i = int(time.time()) % len(self.folder_frames) if not self.collect_target else self.folder_i
            return self.folder_frames[i][1]
        return None

    # ---- learning / inspection --------------------------------------------------
    def start_learning(self):
        if self.next_frame() is None:
            return self.say("No source.")
        self.inspecting = False
        self.collect, self.collect_target = [], max(2, int(self.shots.get()))
        self.say(f"Collecting {self.collect_target} defect-free frames...")

    def _fit(self, frames):
        try:
            self.ins = Inspector(self.kind.get(), float(self.roi.get()),
                                 webhook=self.webhook.get().strip() or None)
            info = self.ins.learn(frames, self.fabric.get().strip() or "fabric")
            self.root.after(0, lambda: (self.profile.configure(values=list_profiles()),
                                        self.say(f"Learned '{info['fabric']}' with {info['detector']} "
                                                 f"from {info['shots']} shots in {info['seconds']:.1f}s; "
                                                 f"threshold {info['threshold']:.3f}.")))
        except Exception as e:
            self.root.after(0, lambda: self.say(f"Learning failed: {e}"))

    def load_profile(self):
        try:
            self.ins = Inspector(roi=float(self.roi.get()), webhook=self.webhook.get().strip() or None)
            self.ins.load_profile(self.profile.get())
            self.say(f"Loaded profile '{self.profile.get()}' ({self.ins.det.name}).")
        except Exception as e:
            self.say(f"Cannot load profile: {e}")

    def toggle_inspect(self):
        if self.ins.det is None:
            return self.say("Learn or load a fabric first.")
        self.ins.webhook = self.webhook.get().strip() or None
        self.inspecting = not self.inspecting
        self.btn_inspect.configure(text="Stop inspection" if self.inspecting else "Start inspection")

    def _inspect(self, frame):
        try:
            r = self.ins.inspect(frame)
            self.root.after(0, lambda: self.show_result(r))
        finally:
            self.busy = False

    def show_result(self, r):
        self.verdict.configure(text="DEFECT" if r["defect"] else "OK",
                               bg="#d62728" if r["defect"] else "#2ca02c", fg="white")
        self.info.configure(text=f"score {r['score']:.3f} / thr {r['threshold']:.3f}\n"
                                 f"area {100 * r['defect_area_frac']:.2f}%   {r['ms']:.0f} ms")
        self.img = _photo(r["overlay"]); self.canvas.configure(image=self.img)

    # ---- main loop --------------------------------------------------------------
    def tick(self):
        f = self.next_frame()
        if f is not None:
            self.frame = f
            if self.collect_target:
                now = time.time()
                if now - self.last_collect > 0.4:
                    self.collect.append(f.copy()); self.last_collect = now
                    self.folder_i += 1
                    if self.folder_frames:
                        self.folder_i %= len(self.folder_frames)
                    self.say(f"Collected {len(self.collect)}/{self.collect_target}")
                if len(self.collect) >= self.collect_target:
                    frames, self.collect_target = self.collect, 0
                    self.say("Fitting model...")
                    threading.Thread(target=self._fit, args=(frames,), daemon=True).start()
            if self.inspecting:
                if not self.busy:
                    self.busy = True
                    threading.Thread(target=self._inspect, args=(f.copy(),), daemon=True).start()
            else:
                self.img = _photo(center_square(f, float(self.roi.get() or 1)))
                self.canvas.configure(image=self.img)
        self.root.after(30, self.tick)

    def say(self, msg):
        self.status.configure(text=msg)


def run():
    root = tk.Tk()
    App(root)
    root.mainloop()
