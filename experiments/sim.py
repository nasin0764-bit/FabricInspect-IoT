"""Trace-driven discrete-event simulation of FabricInspect-IoT.

Each station (texture i mod 5) captures a frame every 1/f s and scores it on its own
single core (service time drawn from the measured station trace). Under the gate a
frame is offloaded with the measured per-texture probability; the payload (drawn from
the measured JPEG sizes + MQTT/TCP/IP overhead) is serialised over the station's own
uplink (rate B) and reaches the edge after the one-way delay d. The edge server is a
single batching server: whenever idle it takes up to b_max queued frames and serves
them in a time drawn from the measured trace for that batch size. Results return after
d. A constant c (messaging/serialisation overhead, calibrated on the live testbed) is
added per offloaded frame.
"""
import heapq
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
LINKS = {"wifi": (20e6, 0.004), "constrained": (1e6, 0.040)}
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
HDR = 60  # bytes of MQTT + TCP/IP header per offload (approx.)


def load_traces():
    t = json.load(open(os.path.join(OUT, "traces.json")))
    return dict(station=np.array(t["station_ms"]) / 1e3,
                edge={int(b): np.array(v) / 1e3 for b, v in t["edge_ms"].items()},
                jpeg={c: np.array(v) for c, v in t["jpeg_bytes"].items()},
                offload=t["offload_p"], valid=t.get("valid"))


def simulate(tr, n, policy, link, fps=5.0, seconds=60.0, b_max=8, overhead=0.0, edge_scale=1.0,
             seed=0):
    rng = np.random.default_rng(seed)
    rate, d = LINKS[link]
    T = 1.0 / fps
    ev = []   # (time, seq, kind, payload)
    k = 0

    def push(t, kind, data):
        nonlocal k
        k += 1
        heapq.heappush(ev, (t, k, kind, data))

    frames = []   # [arrival, done_time or None, offloaded, up_bytes]
    st_free = np.zeros(n)
    up_free = np.zeros(n)
    phase = rng.uniform(0, T, n)
    nf = int(seconds * fps)
    for i in range(n):
        cat = TEXTURES[i % 5]
        p_off = {"station-only": 0.0, "edge-only": 1.0}.get(policy, tr["offload"][cat])
        for j in range(nf):
            a = phase[i] + j * T
            local = 0.0 if policy == "edge-only" else rng.choice(tr["station"])
            start = max(a, st_free[i]); done = start + local; st_free[i] = done
            idx = len(frames)
            if rng.random() < p_off:
                nb = int(rng.choice(tr["jpeg"][cat])) + HDR
                tx0 = max(done, up_free[i]); tx1 = tx0 + 8 * nb / rate; up_free[i] = tx1
                frames.append([a, None, True, nb])
                push(tx1 + d + overhead / 2, "arrive", idx)
            else:
                frames.append([a, done, False, 0])
    # edge server
    queue, busy_until, busy, served_batches = [], 0.0, 0.0, []
    t_end_window = phase.max() + nf * T

    while ev:
        t, _, kind, data = heapq.heappop(ev)
        if kind == "arrive":
            queue.append(data)
            if t >= busy_until:
                push(t, "start", None)
        elif kind == "start":
            if not queue or t < busy_until:
                continue
            batch, queue = queue[:b_max], queue[b_max:]
            svc = edge_scale * rng.choice(tr["edge"][len(batch)])
            busy_until = t + svc
            busy += max(0.0, min(busy_until, t_end_window) - max(t, 0.0)) if t < t_end_window else 0.0
            served_batches.append(len(batch))
            for idx in batch:
                frames[idx][1] = busy_until + d + overhead / 2
            push(busy_until, "start", None)
    lat = np.array([(f[1] - f[0]) * 1000 for f in frames])
    off = np.array([f[2] for f in frames])
    return dict(policy=policy, link=link, stations=n, seed=seed, offload=float(off.mean()),
                lat_p50=float(np.median(lat)), lat_p95=float(np.quantile(lat, .95)),
                deadline_miss=float(np.mean(lat > 1000 * T)),
                uplink_kbps_per_station=float(np.mean([f[3] for f in frames]) * fps * 8 / 1e3),
                edge_util=float(busy / t_end_window),
                mean_batch=float(np.mean(served_batches)) if served_batches else 0.0)


def sweep(overhead, edge_scale=1.0, seeds=10, ns=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64)):
    tr = load_traces()
    rows = []
    for link in LINKS:
        for pol in ("station-only", "cascade", "edge-only"):
            for n in ns:
                rs = [simulate(tr, n, pol, link, overhead=overhead, edge_scale=edge_scale, seed=s)
                      for s in range(seeds)]
                row = dict(policy=pol, link=link, stations=n, edge_scale=edge_scale, seeds=seeds)
                for key in ("offload", "lat_p50", "lat_p95", "deadline_miss", "uplink_kbps_per_station",
                            "edge_util", "mean_batch"):
                    v = np.array([r[key] for r in rs])
                    row[key], row[key + "_sd"] = float(v.mean()), float(v.std())
                rows.append(row)
    return rows


if __name__ == "__main__":
    import sys
    ovh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    out = {"overhead_s": ovh, "nominal": sweep(ovh), "edge_x2": sweep(ovh, edge_scale=2.0)}
    json.dump(out, open(os.path.join(OUT, "sim.json"), "w"), indent=1)
    for r in out["nominal"]:
        print(f"{r['link']:11s} {r['policy']:13s} N={r['stations']:<3d} p95={r['lat_p95']:.0f} "
              f"miss={100 * r['deadline_miss']:.1f}% util={r['edge_util']:.2f}")
