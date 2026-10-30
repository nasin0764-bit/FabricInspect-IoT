"""FabricInspect benchmark on the MVTec AD texture categories.

Usage:  python experiments/run_benchmark.py [fewshot|threshold|robust|latency|ablation|all]
Writes JSON to results/.
"""
import json
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import cv2
import numpy as np
import torch

import data
import metrics
from fabricinspect.detectors import GaborGaussian, GGConfig, PaDiM, PatchCoreLite

torch.set_num_threads(8)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
METHODS = {"Gabor-Gaussian": GaborGaussian, "PatchCore-lite": PatchCoreLite, "PaDiM": PaDiM}
SHOTS = [1, 2, 5, 10, 20, 50, None]   # None = full training set
SEEDS = [0, 1, 2]


def subsets(n, k, seed):
    if k is None or k >= n:
        return list(range(n))
    return sorted(np.random.default_rng(seed).choice(n, k, replace=False).tolist())


def evaluate(det, test_embs, test):
    maps = [det.map_from_embed(e) for e in test_embs]
    lab = [t[2] for t in test]
    msk = [t[1] for t in test]
    return {"img_auroc": metrics.image_auroc(lab, [m.max() for m in maps]),
            "px_auroc": metrics.pixel_auroc(msk, maps),
            "aupro": metrics.aupro(msk, maps)}, maps


def fewshot():
    res = []
    for cat in data.TEXTURES:
        test = data.test_set(cat)
        train = data.load_train(cat)
        for mname, M in METHODS.items():
            det = M()
            tr_e = det.embed(train)
            te_e = det.embed([t[0] for t in test])
            for k in SHOTS:
                for seed in (SEEDS if k is not None else [0]):
                    idx = subsets(len(train), k, seed)
                    t0 = time.perf_counter()
                    det.fit_embeds([tr_e[i] for i in idx])
                    fit_s = time.perf_counter() - t0
                    r, _ = evaluate(det, te_e, test)
                    r.update(cat=cat, method=mname, k=len(idx), seed=seed, fit_s=fit_s)
                    res.append(r)
                    print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "fewshot.json"), "w"), indent=1)


def threshold():
    """Label-free LOO thresholds: FPR on good test images, recall, F1."""
    res = []
    for cat in data.TEXTURES:
        test = data.test_set(cat)
        train = data.load_train(cat)
        lab = np.array([t[2] for t in test])
        for mname in ("Gabor-Gaussian", "PatchCore-lite"):
            det = METHODS[mname]()
            tr_e = det.embed(train)
            te_e = det.embed([t[0] for t in test])
            for k in (5, 10, 20):
                for seed in SEEDS:
                    sub = [tr_e[i] for i in subsets(len(train), k, seed)]
                    det.calibrate(sub)
                    det.fit_embeds(sub)
                    s = np.array([det.map_from_embed(e).max() for e in te_e])
                    base = det.loo_scores.max()
                    for margin in (0.0, 0.1, 0.2, 0.3):
                        thr = base * (1 + margin)
                        p, rc, f1 = metrics.f1_at(lab, s, thr)
                        fpr = float((s[lab == 0] > thr).mean())
                        r = dict(cat=cat, method=mname, k=k, seed=seed, margin=margin,
                                 precision=p, recall=rc, f1=f1, fpr=fpr)
                        res.append(r)
                    print(cat, mname, k, seed, "F1@0.1=%.3f" % res[-3]["f1"], flush=True)
    json.dump(res, open(os.path.join(OUT, "threshold.json"), "w"), indent=1)


PERTURB = {
    "none": lambda x: x,
    "bright-20%": lambda x: np.clip(x.astype(np.float32) * 0.8, 0, 255).astype(np.uint8),
    "bright+20%": lambda x: np.clip(x.astype(np.float32) * 1.2, 0, 255).astype(np.uint8),
    "blur(s=2)": lambda x: cv2.GaussianBlur(x, (0, 0), 2.0),
    "noise(s=10)": lambda x: np.clip(x.astype(np.float32) + np.random.default_rng(0).normal(
        0, 10, x.shape), 0, 255).astype(np.uint8),
}


def robust(k=20):
    res = []
    for cat in data.TEXTURES:
        test = data.test_set(cat)
        train = data.load_train(cat, k, seed=0)
        for mname, M in METHODS.items():
            det = M()
            det.fit_embeds(det.embed(train))
            for pname, f in PERTURB.items():
                # perturb at the 256x256 working resolution (perturbing the 1024x1024
                # original would be largely undone by the downscaling)
                te_e = det.embed([f(cv2.resize(t[0], (256, 256), interpolation=cv2.INTER_AREA))
                                  for t in test])
                r, _ = evaluate(det, te_e, test)
                r.update(cat=cat, method=mname, perturb=pname, k=k)
                res.append(r)
                print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "robust.json"), "w"), indent=1)


def ablation(k=20):
    cfgs = {
        "full (L + Lab chroma + Gabor mean/std)": GGConfig(),
        "w/o chroma (grey only)": GGConfig(use_color=False),
        "w/o Gabor std": GGConfig(use_gabor_std=False),
        "w/o Gabor (intensity + chroma stats only)": GGConfig(use_gabor=False),
        "6 orientations": GGConfig(n_orient=6),
        "pool 9 px": GGConfig(pool=9),
        "pool 25 px": GGConfig(pool=25),
    }
    res = []
    for cat in data.TEXTURES:
        test = data.test_set(cat)
        train = data.load_train(cat, k, seed=0)
        for name, cfg in cfgs.items():
            det = GaborGaussian(cfg)
            det.fit_embeds(det.embed(train))
            r, _ = evaluate(det, det.embed([t[0] for t in test]), test)
            r.update(cat=cat, variant=name, k=k,
                     dim=int(det.mean.shape[0]))
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "ablation.json"), "w"), indent=1)


def latency(k=20, reps=30):
    cat = "carpet"
    train = data.load_train(cat, k, seed=0)
    img = data.test_set(cat)[0][0]
    res = []
    for threads in (8, 2):
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
        for mname, M in METHODS.items():
            det = M()
            t0 = time.perf_counter(); det.fit(train, calibrate=False); fit_s = time.perf_counter() - t0
            det.predict(img)  # warm-up
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); det.predict(img); ts.append(time.perf_counter() - t0)
            state = {"Gabor-Gaussian": lambda d: d.mean.nbytes + d.prec.nbytes,
                     "PatchCore-lite": lambda d: d.bank.numel() * 4,
                     "PaDiM": lambda d: d.mu.nbytes + d.prec.nbytes}[mname](det)
            backbone = 0 if mname == "Gabor-Gaussian" else 11.18e6 * 4  # resnet18 params (fp32)
            r = dict(method=mname, threads=threads, k=k, fit_s=fit_s,
                     ms_median=1000 * float(np.median(ts)), ms_p95=1000 * float(np.quantile(ts, .95)),
                     model_state_MB=state / 1e6, backbone_MB=backbone / 1e6)
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "latency.json"), "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("fewshot", fewshot), ("threshold", threshold), ("ablation", ablation),
                     ("robust", robust), ("latency", latency)]:
        if what in (name, "all"):
            fn()
