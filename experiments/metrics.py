"""Evaluation metrics: image/pixel AUROC, AUPRO (FPR<=0.3), F1 at a threshold."""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, maps, max_px=4_000_000, seed=0):
    y = np.concatenate([m.ravel() for m in masks])
    s = np.concatenate([a.ravel() for a in maps])
    if len(y) > max_px:
        idx = np.random.default_rng(seed).choice(len(y), max_px, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))


def aupro(masks, maps, fpr_limit=0.3, n_thr=1000):
    """Per-region overlap (PRO) integrated over FPR in [0, fpr_limit], normalised to [0,1].

    Exact per threshold via sorted scores (no per-threshold image passes).
    """
    regions = []
    neg = []
    for m, a in zip(masks, maps):
        neg.append(a[m == 0])
        lab, n = ndimage.label(m)
        for r in range(1, n + 1):
            regions.append(np.sort(a[lab == r]))
    if not regions:
        return float("nan")
    neg = np.sort(np.concatenate(neg))
    lo = np.quantile(neg, 1 - fpr_limit)
    thrs = np.linspace(neg[-1], lo, n_thr)
    fpr = 1 - np.searchsorted(neg, thrs, "right") / len(neg)
    pro = np.mean([1 - np.searchsorted(r, thrs, "right") / len(r) for r in regions], 0)
    keep = fpr <= fpr_limit
    return float(np.trapz(pro[keep], fpr[keep]) / fpr_limit)


def f1_at(labels, scores, thr):
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) > thr
    tp = (pred & labels).sum(); fp = (pred & ~labels).sum(); fn = (~pred & labels).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(prec), float(rec), float(f1)
