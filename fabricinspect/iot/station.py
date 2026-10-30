"""Station node: runs at each seaming machine.

The station keeps only the tiny Gabor-Gaussian profile (received as a retained
MQTT message) and applies confidence-gated offloading:

    s < alpha * tau_g          -> OK, decided locally
    s > beta  * tau_g          -> DEFECT, decided locally
    otherwise                  -> frame offloaded to the edge server (PatchCore-lite)

policy 'local' = never offload, 'edge' = always offload, 'cascade' = band above.
"""
import json
import threading
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from . import protocol as P
from .netem import Link
from ..engine import center_square


class Station:
    def __init__(self, name, fabric, host="127.0.0.1", port=1883, policy="cascade",
                 alpha=0.8, beta=1.5, link=None, threads=1, jpeg_quality=85, roi=1.0):
        cv2.setNumThreads(threads)
        self.name, self.fabric, self.policy = name, fabric, policy
        self.alpha, self.beta = alpha, beta
        self.link = link or Link()
        self.q, self.roi = jpeg_quality, roi
        self.det = None
        self._got_profile = threading.Event()
        self.pending = {}            # seq -> record awaiting edge result
        self.records = []
        self._lock = threading.Lock()
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"st-{name}")
        self.c.on_connect = self._on_connect
        self.c.on_message = lambda c, u, m: self.link.receive(lambda: self._on_message(m))
        self.c.connect(host, port, keepalive=30)
        self.c.loop_start()

    # ---- MQTT -----------------------------------------------------------------
    def _on_connect(self, c, u, f, rc, p):
        c.subscribe([(P.t_profile(self.fabric, "gabor"), 1), (P.t_result(self.name), 1)])

    def _on_message(self, m):
        if m.topic.endswith("/gabor"):
            self.det = P.profile_from_bytes(m.payload)
            self.profile_bytes = len(m.payload)
            self.profile_at = time.perf_counter()
            self._got_profile.set()
            return
        r = json.loads(m.payload)
        with self._lock:
            rec = self.pending.pop(r["seq"], None)
        if rec is None:
            return
        now = time.perf_counter()
        rec.update(defect=r["defect"], deep_score=r["score"], server_ms=r["server_ms"],
                   batch=r["batch"], latency_ms=1000 * (now - rec["t0"]), done=True)
        self._verdict(rec)

    def wait_profile(self, timeout=30):
        if not self._got_profile.wait(timeout):
            raise TimeoutError("no profile received for " + self.fabric)

    def _verdict(self, rec):
        v = {k: rec[k] for k in ("seq", "defect", "tier", "latency_ms")}
        v["score"] = rec.get("deep_score", rec["score"])
        v["fabric"] = self.fabric
        msg = json.dumps(v).encode()
        self.link.send(len(msg) + 30, lambda: self.c.publish(P.t_verdict(self.name), msg, qos=0))

    # ---- per frame -------------------------------------------------------------
    def process(self, frame, seq, meta=None):
        t0 = time.perf_counter()
        sq = center_square(frame, self.roi)
        if self.policy == "edge":            # thin client: no local model evaluation
            s, tau = float("nan"), self.det.image_threshold
        else:
            s, tau = self.det.predict(sq)["score"], self.det.image_threshold
        local_ms = 1000 * (time.perf_counter() - t0)
        rec = dict(seq=seq, t0=t0, score=s, local_ms=local_ms, tier="station", done=False,
                   **(meta or {}))
        if self.policy == "edge":
            offload = True
        elif self.policy == "local":
            offload = False
        else:
            offload = self.alpha * tau <= s <= self.beta * tau
        if offload:
            rec["tier"] = "edge"
            payload = P.pack_offload({"seq": seq, "fabric": self.fabric, "t": t0},
                                     cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA),
                                     self.q)
            rec["up_bytes"] = len(payload)
            with self._lock:
                self.pending[seq] = rec
            self.link.send(len(payload) + 40,
                           lambda: self.c.publish(P.t_offload(self.name), payload, qos=1))
        else:
            rec.update(defect=bool(s > tau), latency_ms=local_ms, up_bytes=0, done=True)
            self._verdict(rec)
        self.records.append(rec)
        return rec

    def run(self, frames, fps=5.0, metas=None):
        """Replay frames at a fixed rate (camera emulation)."""
        period = 1.0 / fps
        t_next = time.perf_counter()
        for i, f in enumerate(frames):
            now = time.perf_counter()
            if now < t_next:
                time.sleep(t_next - now)
            self.process(f, i, metas[i] if metas else None)
            t_next += period
        deadline = time.perf_counter() + 10
        while self.pending and time.perf_counter() < deadline:
            time.sleep(0.01)
        return self.records

    def run_camera(self, index=0, fps=5.0, show=False):
        """Live operation from a USB/UVC camera until interrupted."""
        cap = cv2.VideoCapture(index)
        period, seq, t_next = 1.0 / fps, 0, time.perf_counter()
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    raise RuntimeError(f"camera {index} returned no frame")
                if time.perf_counter() >= t_next:
                    rec = self.process(f, seq)
                    seq += 1
                    t_next += period
                    if show:
                        print(f"{seq:6d} {rec['tier']:7s} score={rec['score']:.3f} "
                              f"{'DEFECT' if rec.get('defect') else 'ok' if rec['done'] else '...'}")
                    self.records = self.records[-1000:]
        finally:
            cap.release()

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()
