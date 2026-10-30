"""Measure station-model cost on a real device (Raspberry Pi, mini-PC, Android/Termux).

Needs only numpy, scipy, opencv (no torch). Uses MVTec carpet if present, otherwise
synthetic woven textures. Prints and writes results/device_<hostname>.json.

    python experiments/device_bench.py [--frames 100]
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import cv2
import numpy as np

from fabricinspect.detectors import GaborGaussian


def synthetic(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:256, 0:256]
    g = 128 + 50 * np.sin(x / 3.0 + rng.uniform(0, 6)) * np.sin(y / 3.0 + rng.uniform(0, 6))
    return cv2.cvtColor(np.clip(g + rng.normal(0, 6, g.shape), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    try:
        sys.path.insert(0, HERE)
        import data
        train, test = data.load_train("carpet", 10), [t[0] for t in data.test_set("carpet")]
        src = "mvtec-carpet"
    except Exception:
        train, test, src = [synthetic(i) for i in range(10)], [synthetic(100 + i) for i in range(50)], "synthetic"
    t0 = time.perf_counter()
    det = GaborGaussian().fit(train)
    learn_s = time.perf_counter() - t0
    for x in test[:3]:
        det.predict(x)
    ts = []
    for i in range(a.frames):
        t0 = time.perf_counter(); det.predict(test[i % len(test)]); ts.append(time.perf_counter() - t0)
    ts = 1000 * np.array(ts)
    ok, jpg = cv2.imencode(".jpg", cv2.resize(test[0], (256, 256)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    t0 = time.perf_counter()
    for _ in range(50):
        cv2.imencode(".jpg", cv2.resize(test[0], (256, 256)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_ms = 1000 * (time.perf_counter() - t0) / 50
    r = dict(host=socket.gethostname(), machine=platform.machine(), processor=platform.processor(),
             python=platform.python_version(), source=src, learn_k10_s=learn_s,
             predict_ms_median=float(np.median(ts)), predict_ms_p95=float(np.quantile(ts, .95)),
             max_fps=1000 / float(np.median(ts)), jpeg_encode_ms=jpeg_ms)
    print(json.dumps(r, indent=1))
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    json.dump(r, open(os.path.join(HERE, "..", "results", f"device_{r['host']}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
