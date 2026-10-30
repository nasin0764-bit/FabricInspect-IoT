"""Measure service-time traces for the trace-driven simulator (short, speed-checked runs).

Writes results/traces.json:
  station_ms   : per-frame station-model cost, 1 thread (all 5 textures)
  edge_ms[b]   : per-batch edge cost (decode + PatchCore-lite + scoring) for b = 1..8
  jpeg_bytes   : offload payload sizes per texture (256x256 crop, quality 85)
  offload_p    : per-texture offload probability of the a-priori gate (from iot_cascade.json)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import cv2
import numpy as np
import torch

import data
from fabricinspect.detectors import GaborGaussian, PatchCoreLite
from fabricinspect.iot import protocol as P
from run_iot import ALPHA0, BETA0, speed_check_retry

OUT = os.path.join(HERE, "..", "results")


def main():
    pre = speed_check_retry()
    if not pre["ok"]:
        raise SystemExit("host throttled")
    frames, jpeg = {}, {}
    for cat in data.TEXTURES:
        paths, _ = data.test_paths(cat)
        frames[cat] = [cv2.resize(cv2.imread(p), (256, 256), interpolation=cv2.INTER_AREA) for p in paths]
        jpeg[cat] = [len(P.pack_offload({"seq": 0, "fabric": cat, "t": 0.0}, f)) for f in frames[cat]]

    # station: one thread, every texture, 60 frames each
    cv2.setNumThreads(1)
    st = []
    for cat in data.TEXTURES:
        g = GaborGaussian().fit(data.load_train(cat, 10, seed=0), calibrate=False)
        g.predict(frames[cat][0])
        for f in frames[cat][:60]:
            t0 = time.perf_counter(); g.predict(f); st.append(1000 * (time.perf_counter() - t0))
    cv2.setNumThreads(8)

    # edge: decode + embed + score, batch sizes 1..8, 60 batches each, interleaved order
    torch.set_num_threads(8)
    p = PatchCoreLite().fit(data.load_train("carpet", 10, seed=0), calibrate=False)
    pay = [P.pack_offload({"seq": i, "fabric": "carpet", "t": 0.0}, f) for i, f in enumerate(frames["carpet"])]
    p.predict(frames["carpet"][0])
    edge = {b: [] for b in range(1, 9)}
    rng = np.random.default_rng(0)
    for rep in range(60):
        for b in rng.permutation(np.arange(1, 9)):
            batch = [pay[(rep * 8 + j) % len(pay)] for j in range(b)]
            t0 = time.perf_counter()
            embs = p.embed([P.unpack_offload(x)[1] for x in batch])
            [p.map_from_embed(e).max() for e in embs]
            edge[int(b)].append(1000 * (time.perf_counter() - t0))

    cas = json.load(open(os.path.join(OUT, "iot_cascade.json")))["rows"]
    off = {c: float(np.mean([r["offload"] for r in cas if r["cat"] == c and r["policy"] == "cascade"
                             and r["alpha"] == ALPHA0 and r["beta"] == BETA0])) for c in data.TEXTURES}
    post = speed_check_retry(tries=1)
    out = dict(station_ms=st, edge_ms={str(b): v for b, v in edge.items()}, jpeg_bytes=jpeg, offload_p=off,
               speed_pre=pre, speed_post=post, valid=bool(pre["ok"] and post["ok"]))
    json.dump(out, open(os.path.join(OUT, "traces.json"), "w"), indent=1)
    print("station ms median %.1f p95 %.1f" % (np.median(st), np.quantile(st, .95)))
    for b in edge:
        print("edge b=%d median %.1f ms -> %.0f frames/s" % (b, np.median(edge[b]), 1000 * b / np.median(edge[b])))
    print("jpeg kB", {c: round(np.mean(v) / 1e3, 1) for c, v in jpeg.items()}, "offload", off, "valid", out["valid"])


if __name__ == "__main__":
    main()
