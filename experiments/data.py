"""MVTec AD texture loader (train/good, test/*, ground_truth/*)."""
import glob
import os

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "mvtec")
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]


def _read(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)


def train_paths(cat):
    return sorted(glob.glob(os.path.join(ROOT, cat, "train", "good", "*.png")))


def test_paths(cat):
    """(paths, labels) of the test split without loading images."""
    ps = sorted(glob.glob(os.path.join(ROOT, cat, "test", "*", "*.png")))
    return ps, [int(os.path.basename(os.path.dirname(p)) != "good") for p in ps]


def test_set(cat, size=256):
    """Return list of (image_bgr, mask_uint8[size,size], label, defect_type)."""
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, cat, "test", "*"))):
        typ = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "*.png"))):
            img = _read(p)
            if typ == "good":
                m = np.zeros((size, size), np.uint8)
            else:
                mp = os.path.join(ROOT, cat, "ground_truth", typ,
                                  os.path.basename(p)[:-4] + "_mask.png")
                m = cv2.resize(cv2.imread(mp, 0), (size, size), interpolation=cv2.INTER_NEAREST)
                m = (m > 127).astype(np.uint8)
            out.append((img, m, int(typ != "good"), typ))
    return out


def load_train(cat, k=None, seed=0):
    ps = train_paths(cat)
    if k is not None and k < len(ps):
        rng = np.random.default_rng(seed)
        ps = [ps[i] for i in sorted(rng.choice(len(ps), k, replace=False))]
    return [_read(p) for p in ps]
