"""Small always-on-top window showing progress of the live IoT sweeps (reads the logs)."""
import json
import os
import time
import tkinter as tk
from tkinter import ttk

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PARTS = [("Wi-Fi sweep (3 repetitions)", "log_iot_live_wifi.txt", 45),
         ("Constrained-link sweep", "log_iot_live_constrained.txt", 15)]
TOTAL = sum(p[2] for p in PARTS)


def read(name):
    try:
        with open(os.path.join(R, name), encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines()
    except OSError:
        return []


def rows(lines):
    out = []
    for ln in lines:
        if ln.startswith("{"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


root = tk.Tk()
root.title("FabricInspect – experiment progress")
root.attributes("-topmost", True)
root.geometry("460x230")
root.resizable(False, False)
f = ttk.Frame(root, padding=14)
f.pack(fill="both", expand=True)
ttk.Label(f, text="FabricInspect-IoT live experiment", font=("Segoe UI", 11, "bold")).pack(anchor="w")
total_bar = ttk.Progressbar(f, length=430, maximum=TOTAL)
total_bar.pack(pady=(8, 2))
total_lbl = ttk.Label(f, text="")
total_lbl.pack(anchor="w")
part_lbl = ttk.Label(f, text="")
part_lbl.pack(anchor="w", pady=(8, 0))
part_bar = ttk.Progressbar(f, length=430)
part_bar.pack(pady=2)
cur_lbl = ttk.Label(f, text="", foreground="#52514e", wraplength=430, justify="left")
cur_lbl.pack(anchor="w", pady=(6, 0))
first_seen = {}


def tick():
    done_total, now = 0, time.time()
    active = None
    for title, log, n in PARTS:
        rs = rows(read(log))
        done_total += min(len(rs), n)
        if active is None and len(rs) < n:
            active = (title, log, n, rs)
    if done_total and "t0" not in first_seen:
        first_seen["t0"], first_seen["d0"] = now, done_total
    total_bar["value"] = done_total
    eta = ""
    if "t0" in first_seen and done_total > first_seen["d0"]:
        rate = (now - first_seen["t0"]) / (done_total - first_seen["d0"])
        eta = f" – about {int(rate * (TOTAL - done_total) / 60) + 1} min left"
    total_lbl["text"] = f"{done_total} / {TOTAL} configurations ({100 * done_total // TOTAL}%){eta}"
    if active is None:
        part_lbl["text"], cur_lbl["text"] = "All sweeps finished ✓", "This window closes in 30 s."
        part_bar["value"] = part_bar["maximum"] = 1
        root.after(30000, root.destroy)
        return
    title, log, n, rs = active
    part_lbl["text"] = f"{title}: {len(rs)} / {n}"
    part_bar["maximum"], part_bar["value"] = n, len(rs)
    lines = read(log)
    last = [ln for ln in lines if ln.startswith(("speed check", "cooling", "WARNING", "python :"))]
    if rs:
        r = rs[-1]
        cur = (f"Last: {r['policy']}, {r['stations']} stations, rep {r.get('rep', 0) + 1} – "
               f"p95 {r['lat_p95']:.0f} ms, misses {100 * r['deadline_miss']:.0f}%")
    else:
        cur = last[-1][:120] if last else "starting…"
    if last and ("throttled" in last[-1] or "WARNING" in last[-1]):
        cur = "⚠ " + last[-1][:120]
    cur_lbl["text"] = cur
    root.after(5000, tick)


tick()
root.mainloop()
