"""IoT experiments for FabricInspect-IoT.

  python experiments/run_iot.py cascade   # E1 offline: offload rate vs. accuracy (label-free thresholds)
  python experiments/run_iot.py live      # E2 live: broker + edge server + N station processes
  python experiments/run_iot.py profile   # E3 profile size / distribution time
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import numpy as np

OUT = os.path.join(HERE, "..", "results")
ALPHA0, BETA0 = 0.6, 1.5          # default gating band, fixed a priori
K = 10                            # defect-free shots per fabric
PORT = 18850


def _f1(lab, pred):
    lab, pred = np.asarray(lab, bool), np.asarray(pred, bool)
    tp = (lab & pred).sum(); fp = (~lab & pred).sum(); fn = (lab & ~pred).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(f1=float(2 * p * r / max(p + r, 1e-12)), precision=float(p), recall=float(r),
                fpr=float((pred & ~lab).sum() / max((~lab).sum(), 1)))


# ------------------------------------------------------------------ E1
def cascade():
    import cv2
    import data
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    from run_benchmark import subsets
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    betas = [1.0, 1.25, 1.5, 2.0, float("inf")]
    rows, jpeg = [], {}
    for cat in data.TEXTURES:
        test = data.test_set(cat)
        lab = np.array([t[2] for t in test])
        train = data.load_train(cat)
        jpeg[cat] = float(np.mean([len(cv2.imencode(".jpg", cv2.resize(t[0], (256, 256),
                                   interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
                                   for t in test]))
        g, p = GaborGaussian(), PatchCoreLite()
        ge, pe = g.embed(train), p.embed(train)
        gt, pt = g.embed([t[0] for t in test]), p.embed([t[0] for t in test])
        for seed in (0, 1, 2):
            idx = subsets(len(train), K, seed)
            for det, emb in ((g, ge), (p, pe)):
                sub = [emb[i] for i in idx]
                det.calibrate(sub); det.fit_embeds(sub)
            sg = np.array([g.map_from_embed(e).max() for e in gt])
            sp = np.array([p.map_from_embed(e).max() for e in pt])
            tg, tp = g.image_threshold, p.image_threshold
            base = dict(cat=cat, seed=seed)
            rows.append(dict(base, policy="station-only", alpha=None, beta=None, offload=0.0,
                             **_f1(lab, sg > tg)))
            rows.append(dict(base, policy="edge-only", alpha=None, beta=None, offload=1.0,
                             **_f1(lab, sp > tp)))
            for a in alphas:
                for b in betas:
                    off = (sg >= a * tg) & (sg <= b * tg)
                    pred = np.where(off, sp > tp, sg > tg)
                    rows.append(dict(base, policy="cascade", alpha=a, beta=b,
                                     offload=float(off.mean()), **_f1(lab, pred)))
        print(cat, "done", flush=True)
    json.dump({"rows": rows, "jpeg_bytes": jpeg, "k": K, "default": [ALPHA0, BETA0]},
              open(os.path.join(OUT, "iot_cascade.json"), "w"), indent=1)


# ------------------------------------------------------------------ E2
MOSQUITTO = os.environ.get("FI_MOSQUITTO", os.path.join(HERE, "..", "tools", "mosquitto", "mosquitto.exe"))


def _broker_proc(port):
    import asyncio
    from fabricinspect.iot.broker import Broker
    asyncio.run(Broker().serve("127.0.0.1", port))


class BrokerHandle:
    """Mosquitto if available (default), else the bundled minimal broker."""

    def __init__(self, port, ctx):
        import subprocess
        self.kind = "mosquitto" if os.path.exists(MOSQUITTO) else "bundled"
        if self.kind == "mosquitto":
            conf = os.path.join(OUT, f"mosquitto_{port}.conf")
            open(conf, "w").write(f"listener {port} 127.0.0.1\nallow_anonymous true\n"
                                  "max_queued_messages 10000\n")
            self.p = subprocess.Popen([MOSQUITTO, "-c", conf], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        else:
            self.p = ctx.Process(target=_broker_proc, args=(port,), daemon=True)
            self.p.start()
        time.sleep(1.0)
        print("broker:", self.kind, flush=True)

    def stop(self):
        self.p.terminate()


def _station_proc(name, cat, policy, link, fps, n_frames, port, outq, start_at):
    import cv2
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    import data
    from fabricinspect.iot.netem import PROFILES, Link
    from fabricinspect.iot.station import Station
    # memory: keep only the replayed frames, already at the 256x256 working size
    paths, labels = data.test_paths(cat)
    rng = np.random.default_rng(int(name[1:]))
    order = rng.permutation(len(paths))[:n_frames]
    frames = [cv2.resize(cv2.imread(paths[i], cv2.IMREAD_COLOR), (256, 256), interpolation=cv2.INTER_AREA)
              for i in order]
    metas = [{"label": labels[i]} for i in order]
    st = Station(name, cat, port=port, policy=policy, alpha=ALPHA0, beta=BETA0,
                 link=Link(**PROFILES[link]))
    st.wait_profile(60)
    st.det.predict(frames[0])  # warm-up
    while time.time() < start_at:
        time.sleep(0.005)
    recs = st.run(frames, fps=fps, metas=metas)
    st.close()
    outq.put([{k: v for k, v in r.items() if k != "t0"} for r in recs])


# reference = final measurement state (AC power, Windows power mode "Best performance");
# "Balanced" gives ~30 ms edge, a throttled host 145-170 ms on both - both are rejected
REF_STATION_MS, REF_EDGE_MS = 37.0, 17.0
EDGE_TOLERANCE = 1.5


def speed_check(tolerance=1.25):
    """Measure station (1 thread) and edge (b=1) cost; returns dict with ok flag.

    Laptop CPUs throttle (battery, eco mode, heat) by 4-5x; timing results are only
    meaningful when the host runs at its reference speed.
    """
    import cv2
    import torch
    import data
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    tr = data.load_train("carpet", 10, seed=0)
    x = [cv2.resize(t, (256, 256), interpolation=cv2.INTER_AREA) for t in data.load_train("carpet", 20, seed=1)]
    n_cv = cv2.getNumThreads(); cv2.setNumThreads(1)
    g = GaborGaussian().fit(tr, calibrate=False); g.predict(x[0])
    ts = []
    for im in x:
        t0 = time.perf_counter(); g.predict(im); ts.append(time.perf_counter() - t0)
    cv2.setNumThreads(n_cv)
    torch.set_num_threads(8)
    p = PatchCoreLite().fit(tr, calibrate=False); p.predict(x[0])
    te = []
    for im in x:
        t0 = time.perf_counter(); p.predict(im); te.append(time.perf_counter() - t0)
    st, ed = 1000 * float(np.median(ts)), 1000 * float(np.median(te))
    ok = st <= REF_STATION_MS * tolerance and ed <= REF_EDGE_MS * EDGE_TOLERANCE
    r = dict(station_ms=st, edge_ms=ed, ok=bool(ok), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("speed check:", r, flush=True)
    return r


def speed_check_retry(tries=3, cooldown=60):
    for i in range(tries):
        r = speed_check()
        if r["ok"]:
            return r
        if i < tries - 1:
            print(f"cooling down {cooldown}s before re-check", flush=True)
            time.sleep(cooldown)
    return r


def live():
    pre = speed_check_retry()
    if not pre["ok"] and not os.environ.get("FI_FORCE"):
        raise SystemExit("host is throttled (see speed check) - plug in, set performance mode, let it cool, retry")
    import data
    from fabricinspect.iot.edge import EdgeServer
    from fabricinspect.iot.registry import publish_profile
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    ctx = mp.get_context("spawn")
    bp = BrokerHandle(PORT, ctx)
    srv = EdgeServer(port=PORT, max_batch=8, threads=8)
    sizes = {}
    for cat in data.TEXTURES:
        tr = data.load_train(cat, K, seed=0)
        g, p = GaborGaussian().fit(tr), PatchCoreLite().fit(tr)
        sizes[cat] = publish_profile(cat, g, port=PORT)
        srv.add_model(cat, p)
    for cat in data.TEXTURES:   # warm-up the edge model
        srv.models[cat].predict(data.load_train(cat, 1)[0])
    results = []
    quick = os.environ.get("FI_QUICK")
    fps, n_frames = 5.0, (10 if quick else 60)
    links = os.environ.get("FI_LINKS", "wifi" if quick else "wifi,constrained").split(",")
    ns = [int(x) for x in os.environ.get("FI_NS", "1,2" if quick else "1,4,8,16,32").split(",")]
    reps = int(os.environ.get("FI_REPS", "1"))
    out_name = os.environ.get("FI_OUT", "iot_live.json")
    for rep, link, policy, n in [(r, l, p, n) for r in range(reps) for l in links
                                 for p in ("station-only", "cascade", "edge-only") for n in ns]:
            if True:
                pol = {"station-only": "local", "edge-only": "edge"}.get(policy, policy)
                # drain: no backlog from an overloaded previous configuration may leak in
                t_drain = time.time()
                while not srv.q.empty() and time.time() - t_drain < 600:
                    time.sleep(0.2)
                time.sleep(2.0)
                q = ctx.Queue()
                start_at = time.time() + 8 + 0.25 * n
                ps = [ctx.Process(target=_station_proc,
                                  args=(f"s{i:02d}", data.TEXTURES[i % 5], pol, link, fps,
                                        n_frames, PORT, q, start_at)) for i in range(n)]
                [p.start() for p in ps]
                # steady-state window = the capture period shared by all stations
                w0 = time.perf_counter() + (start_at - time.time())
                w1 = w0 + n_frames / fps
                recs = [r for _ in ps for r in q.get(timeout=300)]
                util = srv.utilisation(w0, w1)
                batches = srv.batches_in(w0, w1)
                [p.join() for p in ps]
                lat = np.array([r["latency_ms"] for r in recs if r["done"]])
                done = [r for r in recs if r["done"]]
                row = dict(rep=rep, link=link, policy=policy, stations=n, fps=fps, frames=len(recs),
                           completed=len(done) / len(recs),
                           offload=float(np.mean([r["tier"] == "edge" for r in recs])),
                           lat_mean=float(lat.mean()), lat_p50=float(np.median(lat)),
                           lat_p95=float(np.quantile(lat, .95)), lat_p99=float(np.quantile(lat, .99)),
                           # unanswered frames count as misses
                           deadline_miss=float(((lat > 1000 / fps).sum() + len(recs) - len(done)) / len(recs)),
                           local_ms=float(np.median([r["local_ms"] for r in recs])),
                           up_bytes_per_frame=float(np.mean([r["up_bytes"] for r in recs])),
                           uplink_kbps_per_station=float(np.mean([r["up_bytes"] for r in recs]) * fps * 8 / 1e3),
                           edge_util=util, mean_batch=float(np.mean(batches)) if batches else 0.0,
                           **_f1([r["label"] for r in done], [r["defect"] for r in done]))
                results.append(row)
                print(json.dumps(row), flush=True)
                json.dump({"rows": results, "profile_bytes": sizes, "broker": bp.kind, "speed_pre": pre},
                          open(os.path.join(OUT, out_name), "w"), indent=1)
    bp.stop()
    post = speed_check_retry(tries=1)
    json.dump({"rows": results, "profile_bytes": sizes, "broker": bp.kind, "speed_pre": pre,
               "speed_post": post, "valid": bool(pre["ok"] and post["ok"])},
              open(os.path.join(OUT, out_name), "w"), indent=1)
    if not post["ok"]:
        print("WARNING: host throttled by the end of the sweep - results invalid", flush=True)


# ------------------------------------------------------------------ E3
def _profile_sub(port, topic, outq, t_pub):
    import paho.mqtt.client as mqtt
    from fabricinspect.iot.netem import PROFILES, Link
    import threading
    ev = threading.Event()
    res = {}
    link = Link(**PROFILES[os.environ["FI_LINK"]])

    def on_msg(c, u, m):
        # downlink: emulate the same bit-rate as uplink for the profile transfer
        def done():
            res["t"] = time.time(); ev.set()
        link.send(len(m.payload), done)
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.on_connect = lambda c, u, f, rc, p: c.subscribe(topic, 1)
    c.on_message = on_msg
    c.connect("127.0.0.1", port); c.loop_start()
    ev.wait(120)
    outq.put(res["t"] - t_pub.value)


def profile():
    import data
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    from fabricinspect.iot import protocol as P
    from fabricinspect.iot.registry import publish_profile
    ctx = mp.get_context("spawn")
    bp = BrokerHandle(PORT + 1, ctx)
    rows = []
    for cat in data.TEXTURES:
        tr = data.load_train(cat, K, seed=0)
        t0 = time.perf_counter(); g = GaborGaussian().fit(tr); tg = time.perf_counter() - t0
        t0 = time.perf_counter(); p = PatchCoreLite().fit(tr); tp = time.perf_counter() - t0
        rows.append(dict(cat=cat, gabor_bytes=len(P.profile_bytes(g)), patchcore_bytes=len(P.profile_bytes(p)),
                         gabor_learn_s=tg, patchcore_learn_s=tp))
        print(rows[-1], flush=True)
    # distribution time of the station profile to N stations over each link
    dist = []
    tr = data.load_train("carpet", K, seed=0)
    g, p = GaborGaussian().fit(tr), PatchCoreLite().fit(tr)
    for kind, det in (("gabor", g), ("patchcore", p)):
        payload = P.profile_bytes(det)
        for link in ("wifi", "constrained"):
            for n in (1, 8, 32):
                os.environ["FI_LINK"] = link
                topic = P.t_profile(f"dist{kind}{link}{n}", kind)
                q, t_pub = ctx.Queue(), ctx.Value("d", 0.0)
                ps = [ctx.Process(target=_profile_sub, args=(PORT + 1, topic, q, t_pub)) for _ in range(n)]
                [x.start() for x in ps]; time.sleep(6 + 0.2 * n)
                import paho.mqtt.client as mqtt
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); c.connect("127.0.0.1", PORT + 1); c.loop_start()
                t_pub.value = time.time()
                c.publish(topic, payload, qos=1, retain=True).wait_for_publish(30)
                ts = [q.get(timeout=180) for _ in ps]
                [x.join() for x in ps]; c.loop_stop(); c.disconnect()
                dist.append(dict(kind=kind, link=link, stations=n, bytes=len(payload),
                                 t_mean_s=float(np.mean(ts)), t_max_s=float(np.max(ts))))
                print(dist[-1], flush=True)
    json.dump({"sizes": rows, "distribution": dist}, open(os.path.join(OUT, "iot_profile.json"), "w"), indent=1)
    bp.stop()


# ------------------------------------------------------------------ edge throughput mu(b)
def edgebench(reps=20):
    import cv2
    import torch
    import data
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    from fabricinspect.iot import protocol as P
    torch.set_num_threads(8)
    tr = data.load_train("carpet", K, seed=0)
    test = [t[0] for t in data.test_set("carpet")]
    p = PatchCoreLite().fit(tr)
    payloads = [P.pack_offload({"seq": i, "fabric": "carpet", "t": 0}, cv2.resize(x, (256, 256),
                interpolation=cv2.INTER_AREA)) for i, x in enumerate(test[:16])]
    rows = []
    for b in (1, 2, 4, 8, 16):
        ts = []
        for r in range(reps):
            batch = payloads[(r * b) % 16:(r * b) % 16 + b] or payloads[:b]
            batch = (batch + payloads)[:b]
            t0 = time.perf_counter()
            imgs = [P.unpack_offload(x)[1] for x in batch]
            embs = p.embed(imgs)
            _ = [p.map_from_embed(e).max() for e in embs]
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        rows.append(dict(batch=b, batch_ms=1000 * t, per_frame_ms=1000 * t / b, mu_fps=b / t))
        print(rows[-1], flush=True)
    # station compute on one core
    cv2.setNumThreads(1)
    g = GaborGaussian().fit(tr)
    for x in test[:3]:
        g.predict(x)
    ts = []
    for x in test[:40]:
        t0 = time.perf_counter(); g.predict(x); ts.append(time.perf_counter() - t0)
    st = dict(station_ms_median=1000 * float(np.median(ts)), station_ms_p95=1000 * float(np.quantile(ts, .95)))
    print(st)
    json.dump({"edge": rows, "station": st}, open(os.path.join(OUT, "iot_edgebench.json"), "w"), indent=1)


if __name__ == "__main__":
    {"cascade": cascade, "live": live, "profile": profile, "edgebench": edgebench,
     "speedcheck": speed_check}[sys.argv[1]]()

