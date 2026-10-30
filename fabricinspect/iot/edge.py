"""Edge server: verifies offloaded frames with PatchCore-lite (micro-batched).

    python -m fabricinspect.iot.edge --host 127.0.0.1 --profiles ~/FabricInspect/profiles
"""
import argparse
import glob
import json
import os
import queue
import threading
import time

import paho.mqtt.client as mqtt

from . import protocol as P
from .. import detectors as D


class EdgeServer:
    def __init__(self, host="127.0.0.1", port=1883, max_batch=8, threads=8):
        import torch
        torch.set_num_threads(threads)
        self.models = {}             # fabric -> PatchCoreLite
        self.q = queue.Queue()
        self.max_batch = max_batch
        self.busy = 0.0
        self.served = 0
        self.t_start = time.perf_counter()
        self.batches = []
        self.log = []
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge")
        self.c.on_connect = lambda c, u, f, rc, p: c.subscribe(P.t_offload("+"), 1)
        self.c.on_message = lambda c, u, m: self.q.put((time.perf_counter(), m.topic, m.payload))
        self.c.connect(host, port, keepalive=30)
        self.c.loop_start()
        threading.Thread(target=self._worker, daemon=True).start()

    def add_model(self, fabric, det):
        self.models[fabric] = det

    def load_profiles(self, folder):
        for p in glob.glob(os.path.join(folder, "*.npz")):
            det = D.load(p)
            if isinstance(det, D.PatchCoreLite):
                self.add_model(os.path.basename(p)[:-4], det)

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.max_batch:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            t0 = time.perf_counter()
            dec = [(ta, P.station_of(t), *P.unpack_offload(pl)) for ta, t, pl in items]
            by_fabric = {}
            for d in dec:
                by_fabric.setdefault(d[2]["fabric"], []).append(d)
            for fabric, ds in by_fabric.items():
                det = self.models.get(fabric)
                if det is None:
                    continue
                embs = det.embed([d[3] for d in ds])
                for (ta, st, h, _), e in zip(ds, embs):
                    s = float(det.map_from_embed(e).max())
                    msg = {"seq": h["seq"], "score": s, "defect": bool(s > det.image_threshold),
                           "threshold": det.image_threshold, "batch": len(items),
                           "server_ms": 1000 * (time.perf_counter() - ta)}
                    self.c.publish(P.t_result(st), json.dumps(msg), qos=1)
            t1 = time.perf_counter()
            self.busy += t1 - t0
            self.served += len(items)
            self.batches.append(len(items))
            self.log.append((t0, t1, len(items)))      # for exact windowed utilisation

    def utilisation(self, t_from=None, t_to=None):
        """Busy fraction; with a window, computed exactly from per-batch timestamps."""
        if t_from is None:
            return self.busy / max(time.perf_counter() - self.t_start, 1e-9)
        busy = sum(max(0.0, min(b, t_to) - max(a, t_from)) for a, b, _ in list(self.log))
        return busy / max(t_to - t_from, 1e-9)

    def batches_in(self, t_from, t_to):
        return [n for a, b, n in list(self.log) if t_from <= b <= t_to]

    def reset_stats(self):
        self.busy, self.served, self.batches = 0.0, 0, []
        self.t_start = time.perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--profiles", default=os.path.join(os.path.expanduser("~"), "FabricInspect", "profiles"))
    ap.add_argument("--max-batch", type=int, default=8)
    a = ap.parse_args()
    srv = EdgeServer(a.host, a.port, a.max_batch)
    srv.load_profiles(a.profiles)
    print("edge server ready; fabrics:", sorted(srv.models))
    while True:
        time.sleep(10)
        print(json.dumps({"served": srv.served, "util": round(srv.utilisation(), 3),
                          "queue": srv.q.qsize()}))
