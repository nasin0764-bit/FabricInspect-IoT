"""Pictures for the quick guide, produced by the real app engine on MVTec AD images."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import fabricinspect.engine as E

OUT = os.path.join(HERE, "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"
OK_C, BAD_C = "#0ca30c", "#d03b3b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 200, "savefig.bbox": "tight"})

# keep the user's real profile/log folders untouched
tmp = tempfile.mkdtemp()
E.PROFILE_DIR, E.LOG_DIR = os.path.join(tmp, "profiles"), os.path.join(tmp, "logs")


def rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def learn_inspect(cat, defect_type, name):
    train = data.load_train(cat, 10, seed=0)
    ins = E.Inspector("patchcore")
    info = ins.learn(train, cat)
    test = data.test_set(cat)
    good = next(t for t in test if t[3] == "good")
    bad = next(t for t in test if t[3] == defect_type)
    rg, rb = ins.inspect(good[0]), ins.inspect(bad[0])

    fig = plt.figure(figsize=(7.2, 4.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.5], hspace=0.04)
    top.suptitle(f"① LEARN: show the camera 10 defect-free frames of the fabric ({cat}) – "
                 f"learned in {info['seconds']:.1f} s, no labels needed", fontsize=8.5, fontweight="bold", color=INK)
    for j, ax in enumerate(top.subplots(1, 6)):
        ax.imshow(rgb(cv2.resize(train[j], (160, 160)))); ax.axis("off")
    bottom.suptitle("② INSPECT: every new frame gets a verdict and a heat-map\n"
                    "(red outline = area that does not look like the learned fabric)",
                    fontsize=8.5, fontweight="bold", color=INK)
    axes = bottom.subplots(1, 2, gridspec_kw=dict(top=0.74, bottom=0.02))
    for ax, (r, title, col) in zip(axes, ((rg, "OK", OK_C), (rb, "DEFECT", BAD_C))):
        ax.imshow(rgb(r["overlay"])); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(3)
        ax.set_title(f"{title}\nscore {r['score']:.2f}  (threshold {r['threshold']:.2f}),  {r['ms']:.0f} ms",
                     fontsize=8, color=col, fontweight="bold")
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)
    print(name, info["seconds"], rg["score"], rb["score"], rg["defect"], rb["defect"])


def gallery(name):
    picks = [("carpet", "thread"), ("carpet", "hole"), ("leather", "cut"), ("leather", "color"),
             ("wood", "scratch"), ("tile", "crack")]
    fig, axs = plt.subplots(2, 6, figsize=(7.2, 2.7))
    cache = {}
    for j, (cat, typ) in enumerate(picks):
        if cat not in cache:
            ins = E.Inspector("patchcore"); ins.learn(data.load_train(cat, 10, seed=0), cat)
            cache[cat] = (ins, data.test_set(cat))
        ins, test = cache[cat]
        t = next(x for x in test if x[3] == typ)
        r = ins.inspect(t[0])
        axs[0, j].imshow(rgb(cv2.resize(t[0], (256, 256))))
        axs[1, j].imshow(rgb(r["overlay"]))
        axs[0, j].set_title(f"{cat}: {typ.replace('_', ' ')}", fontsize=7.5)
        verdict = "DEFECT" if r["defect"] else "missed"
        axs[1, j].set_title(verdict, fontsize=7.5, color=BAD_C if r["defect"] else INK2, fontweight="bold")
        for a in (axs[0, j], axs[1, j]):
            a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_ylabel("camera", fontsize=8); axs[1, 0].set_ylabel("app view", fontsize=8)
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


if __name__ == "__main__":
    learn_inspect("carpet", "thread", "guide_learn_inspect.png")
    gallery("guide_gallery.png")
    import shutil
    for f in ("fig_architecture.png", "fig_gate.png", "fig_sim.png"):
        shutil.copy(os.path.join(HERE, "..", "paper", "ieee", "figures", f), os.path.join(OUT, f))
    print(os.listdir(OUT))
