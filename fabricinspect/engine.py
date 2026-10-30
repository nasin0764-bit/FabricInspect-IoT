"""Camera-agnostic inspection engine shared by the GUI and the CLI."""
from __future__ import annotations

import csv
import glob
import json
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

from . import detectors as D

PROFILE_DIR = os.path.join(os.path.expanduser("~"), "FabricInspect", "profiles")
LOG_DIR = os.path.join(os.path.expanduser("~"), "FabricInspect", "logs")


def center_square(frame: np.ndarray, roi: float = 1.0) -> np.ndarray:
    """Central square crop; `roi` (0,1] shrinks it further around the centre."""
    h, w = frame.shape[:2]
    s = int(min(h, w) * roi)
    y, x = (h - s) // 2, (w - s) // 2
    return frame[y:y + s, x:x + s]


def overlay(frame_sq: np.ndarray, amap: np.ndarray, thr: float | None, alpha=0.45) -> np.ndarray:
    """Jet heat-map over the frame; regions above `thr` are outlined in red."""
    img = cv2.resize(frame_sq, amap.shape[::-1])
    hi = thr if thr else float(amap.max()) or 1.0
    norm = np.clip(amap / (hi * 1.5), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    if thr:
        cnts, _ = cv2.findContours((amap > thr).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out


def list_profiles() -> list:
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PROFILE_DIR, "*.npz")))


def read_images(folder: str) -> list:
    ps = sorted(p for e in ("png", "jpg", "jpeg", "bmp", "tif") for p in
                glob.glob(os.path.join(folder, f"*.{e}")))
    return [(p, cv2.imread(p, cv2.IMREAD_COLOR)) for p in ps]


class Inspector:
    """Learn a fabric from defect-free frames, then score frames and log results."""

    def __init__(self, kind: str = "patchcore", roi: float = 1.0, margin: float = 0.10,
                 webhook: str | None = None):
        self.kind, self.roi, self.margin, self.webhook = kind, roi, margin, webhook
        self.det: D.Detector | None = None
        self.fabric = None
        self._log = None

    # ---- learning ---------------------------------------------------------
    def learn(self, frames: list, fabric: str, save: bool = True) -> dict:
        if len(frames) < 2:
            raise ValueError("need at least 2 defect-free frames to calibrate a threshold")
        sq = [center_square(f, self.roi) for f in frames]
        t0 = time.perf_counter()
        self.det = D.DETECTORS[self.kind]().fit(sq, margin=self.margin)
        self.fabric = fabric
        if save:
            os.makedirs(PROFILE_DIR, exist_ok=True)
            D.save(self.det, os.path.join(PROFILE_DIR, fabric + ".npz"))
        return {"fabric": fabric, "detector": self.det.name, "shots": len(frames),
                "seconds": time.perf_counter() - t0, "threshold": self.det.image_threshold,
                "loo_scores": [round(float(s), 3) for s in self.det.loo_scores]}

    def load_profile(self, fabric: str):
        self.det = D.load(os.path.join(PROFILE_DIR, fabric + ".npz"))
        self.kind = next(k for k, v in D.DETECTORS.items() if isinstance(self.det, v))
        self.fabric = fabric

    # ---- inspection ---------------------------------------------------------
    def inspect(self, frame: np.ndarray, source: str = "camera", save_defects: bool = True) -> dict:
        if self.det is None:
            raise RuntimeError("no fabric learned or loaded")
        sq = center_square(frame, self.roi)
        t0 = time.perf_counter()
        r = self.det.predict(sq)
        r["ms"] = 1000 * (time.perf_counter() - t0)
        r["overlay"] = overlay(sq, r["map"], self.det.image_threshold)
        r["threshold"] = self.det.image_threshold
        self._write_log(r, source, save_defects)
        if self.webhook:
            self._notify(r)
        return r

    def _write_log(self, r, source, save_defects):
        os.makedirs(LOG_DIR, exist_ok=True)
        day = datetime.now().strftime("%Y%m%d")
        path = os.path.join(LOG_DIR, f"inspect_{day}.csv")
        new = not os.path.exists(path)
        ts = datetime.now().isoformat(timespec="milliseconds")
        snap = ""
        if save_defects and r["defect"]:
            snap = os.path.join(LOG_DIR, "defects", f"{ts.replace(':', '-')}_{self.fabric}.jpg")
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            cv2.imwrite(snap, r["overlay"])
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "fabric", "detector", "source", "score", "threshold",
                            "defect", "defect_area_frac", "ms", "snapshot"])
            w.writerow([ts, self.fabric, self.det.name, source, f"{r['score']:.4f}",
                        f"{r['threshold']:.4f}", int(r["defect"]),
                        f"{r['defect_area_frac']:.5f}", f"{r['ms']:.1f}", snap])

    def _notify(self, r):
        """POST a small JSON verdict, e.g. to a seaming-machine controller.

        Payload: {"fabric", "defect", "score", "threshold", "defect_area_frac"}.
        The receiving firmware must accept this schema; errors are swallowed so
        a network hiccup never stops inspection.
        """
        body = json.dumps({"fabric": self.fabric, "defect": r["defect"],
                           "score": round(r["score"], 4), "threshold": round(r["threshold"], 4),
                           "defect_area_frac": round(r["defect_area_frac"], 5)}).encode()
        try:
            req = urllib.request.Request(self.webhook, body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=0.5).close()
        except Exception:
            pass
