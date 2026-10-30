"""Build paper figures (PNG, 300 dpi) and tables (Markdown + JSON) from results/*.json."""
import json
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data

R = os.path.join(os.path.dirname(__file__), "..", "results")
P = os.path.join(os.path.dirname(__file__), "..", "paper")
F = os.path.join(P, "figures")
os.makedirs(F, exist_ok=True)

METHODS = ["Gabor-Gaussian", "PatchCore-lite", "PaDiM"]
COL = {"Gabor-Gaussian": "#2a78d6", "PatchCore-lite": "#eb6834", "PaDiM": "#1baf7a"}
MRK = {"Gabor-Gaussian": "o", "PatchCore-lite": "s", "PaDiM": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 300,
    "savefig.bbox": "tight", "legend.frameon": False,
})
CATS = data.TEXTURES


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def agg(rows, keys, metric):
    """mean, std over seeds of `metric` grouped by `keys`."""
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in g.items()}


tables = {}


# ---------------------------------------------------------------- few-shot
def fig_fewshot(rows):
    ks = sorted({r["k"] for r in rows if r["cat"] == "carpet"})
    for metric, fname, ylab in [("img_auroc", "fig_fewshot_image.png", "Image AUROC"),
                                ("aupro", "fig_fewshot_aupro.png", "AUPRO (FPR≤0.3)")]:
        fig, axs = plt.subplots(1, 6, figsize=(7.2, 2.0), sharey=True)
        for ax, cat in zip(axs, CATS + ["mean"]):
            for m in METHODS:
                if cat == "mean":
                    # mean over categories per seed, then mean/std over seeds
                    per = {}
                    for r in rows:
                        if r["method"] == m:
                            kk = "full" if r["k"] == max(x["k"] for x in rows if x["cat"] == r["cat"]) else r["k"]
                            per.setdefault((kk, r["seed"]), []).append(r[metric])
                    xs = [k for k in ks if k <= 50] + ["full"]
                    mu = [np.mean([np.mean(per[(k, s)]) for s in range(3) if (k, s) in per]) for k in xs]
                    sd = [np.std([np.mean(per[(k, s)]) for s in range(3) if (k, s) in per]) for k in xs]
                    x = [k if k != "full" else 230 for k in xs]
                else:
                    a = agg([r for r in rows if r["cat"] == cat and r["method"] == m], ["k"], metric)
                    ks_ = sorted(a); x = [k[0] for k in ks_]
                    mu = [a[k][0] for k in ks_]; sd = [a[k][1] for k in ks_]
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(x, mu - sd, mu + sd, color=COL[m], alpha=0.15, lw=0)
                ax.plot(x, mu, color=COL[m], lw=1.6, marker=MRK[m], ms=3.5, label=m)
            ax.set_xscale("log"); ax.set_title(cat, color=INK, fontsize=8)
            ax.set_xticks([1, 5, 20, 100]); ax.set_xticklabels(["1", "5", "20", "100"])
            ax.minorticks_off()
        fig.supxlabel("defect-free shots k", fontsize=8, y=-0.04)
        axs[0].set_ylabel(ylab)
        axs[0].set_ylim(0.4 if metric == "img_auroc" else 0.2, 1.0)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
        fig.savefig(os.path.join(F, fname)); plt.close(fig)

    # table: per category at k=10 and full set
    out = []
    for kk in (10, "full"):
        for cat in CATS:
            row = {"k": kk, "cat": cat}
            kmax = max(r["k"] for r in rows if r["cat"] == cat)
            k = kmax if kk == "full" else kk
            for m in METHODS:
                for met in ("img_auroc", "px_auroc", "aupro"):
                    a = agg([r for r in rows if r["cat"] == cat and r["method"] == m and r["k"] == k],
                            ["k"], met)[(k,)]
                    row[f"{m}|{met}"] = a
            out.append(row)
    tables["fewshot"] = out
    # fit time table (median over cats/seeds) by k
    ft = {}
    for m in METHODS:
        for r in rows:
            if r["method"] == m:
                ft.setdefault((m, r["k"] if r["k"] <= 50 else "full"), []).append(r["fit_s"])
    tables["fit_time"] = {f"{m}|{k}": float(np.median(v)) for (m, k), v in ft.items()}


# ---------------------------------------------------------------- threshold
def tab_threshold(rows):
    out = {}
    for r in rows:
        out.setdefault((r["method"], r["k"], r["margin"]), []).append(r)
    t = []
    for (m, k, mg), rs in sorted(out.items()):
        t.append({"method": m, "k": k, "margin": mg,
                  **{x: (float(np.mean([r[x] for r in rs])), float(np.std([r[x] for r in rs])))
                     for x in ("precision", "recall", "f1", "fpr")}})
    tables["threshold"] = t
    # per-category at k=10, margin 0.1
    pc = {}
    for r in rows:
        if r["k"] == 10 and r["margin"] == 0.1:
            pc.setdefault((r["method"], r["cat"]), []).append(r)
    tables["threshold_cat"] = [{"method": m, "cat": c,
                                **{x: float(np.mean([r[x] for r in rs])) for x in ("f1", "fpr", "recall")}}
                               for (m, c), rs in sorted(pc.items())]


# ---------------------------------------------------------------- qualitative
def fig_qualitative(k=10):
    from fabricinspect.detectors import GaborGaussian, PatchCoreLite
    from fabricinspect.engine import overlay
    picks = {"carpet": "thread", "grid": "broken", "leather": "cut", "tile": "crack", "wood": "scratch"}
    fig, axs = plt.subplots(4, 5, figsize=(7.2, 6.0))
    for j, cat in enumerate(CATS):
        train = data.load_train(cat, k, seed=0)
        test = [t for t in data.test_set(cat) if t[3] == picks[cat]]
        img, mask = test[0][0], test[0][1]
        sq = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        panels = [cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)]
        gt = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt, cnts, -1, (255, 255, 0), 2)
        panels.append(gt)
        for D in (GaborGaussian, PatchCoreLite):
            d = D().fit(train)
            r = d.predict(img)
            ov = overlay(sq, r["map"], d.image_threshold)
            panels.append(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        for i, p in enumerate(panels):
            axs[i, j].imshow(p); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].grid(False)
            for s in axs[i, j].spines.values():
                s.set_visible(False)
        axs[0, j].set_title(f"{cat} / {picks[cat]}", fontsize=8, color=INK)
    for i, lab in enumerate(["input", "ground truth", "Gabor-Gaussian", "PatchCore-lite"]):
        axs[i, 0].set_ylabel(lab, fontsize=8, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(F, "fig_qualitative.png")); plt.close(fig)


# ---------------------------------------------------------------- system diagram
def fig_system():
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    boxes = [
        (1, 11, 12, 9, "USB camera\nover\nseam line"),
        (17, 18, 15, 9, "LEARN\n5–20 defect-free\nframes"),
        (17, 3, 15, 9, "INSPECT\nlive frames"),
        (36, 11, 17, 9, "Detector\nPatchCore-lite or\nGabor-Gaussian"),
        (57, 18, 19, 9, "Fabric profile\n+ LOO threshold"),
        (57, 3, 19, 9, "OK / DEFECT verdict,\nheat-map, CSV log,\nsnapshots"),
        (80, 3, 19, 9, "Seaming machine\n(optional webhook)"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                                       fc="#f0efec", ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7, color=INK)
    arr = dict(arrowstyle="-|>", color=INK2, lw=0.9)
    for a, b in [((13.5, 17.5), (16.5, 22)), ((13.5, 13.5), (16.5, 8)), ((32.5, 22), (35.5, 17)),
                 ((32.5, 8), (35.5, 13)), ((53.5, 17), (56.5, 22)), ((53.5, 13), (56.5, 8)),
                 ((76.5, 7.5), (79.5, 7.5)), ((66.5, 17.5), (66.5, 12.5))]:
        ax.annotate("", b, a, arrowprops=arr)
    fig.savefig(os.path.join(F, "fig_system.png")); plt.close(fig)


# ================================================================ LaTeX tables
T = os.path.join(P, "ieee", "tables")
os.makedirs(T, exist_ok=True)


def _w(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf-8").write(s)


def _bold_best(vals, fmt="{:.3f}", hi=True):
    best = max(vals) if hi else min(vals)
    return [("\\textbf{" + fmt.format(v) + "}") if abs(v - best) < 1e-12 else fmt.format(v) for v in vals]


def tex_k10(rows, k=10):
    lines = []
    means = {m: {x: [] for x in ("img_auroc", "px_auroc", "aupro")} for m in METHODS}
    for cat in CATS + ["mean"]:
        cells = []
        for met in ("img_auroc", "px_auroc", "aupro"):
            vals = []
            for m in METHODS:
                if cat == "mean":
                    v = np.mean(means[m][met])
                else:
                    v = np.mean([r[met] for r in rows if r["cat"] == cat and r["method"] == m and r["k"] == k])
                    means[m][met].append(v)
                vals.append(v)
            cells += _bold_best(vals)
        name = "\\textit{Mean}" if cat == "mean" else cat
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\" + (" \\midrule" if cat == "wood" else ""))
    _w("tab_k10", r"""\begin{table*}[t]
\centering
\caption{Detection and localisation with $k=10$ defect-free shots (mean of three random draws). GG: Gabor--Gaussian (station), PC: PatchCore-lite (edge), PD: PaDiM. Best per metric in bold.}
\label{tab:k10}
\footnotesize
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& \multicolumn{3}{c}{Image AUROC} & \multicolumn{3}{c}{Pixel AUROC} & \multicolumn{3}{c}{AUPRO} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Category & GG & PC & PD & GG & PC & PD & GG & PC & PD \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tex_threshold(th):
    def m_(meth, k, mg, x):
        return np.mean([r[x] for r in th if r["method"] == meth and r["k"] == k and r["margin"] == mg])
    lines = []
    for meth, short in (("Gabor-Gaussian", "GG (station)"), ("PatchCore-lite", "PC (edge)")):
        first = True
        for k, mg in ((10, 0.0), (10, 0.1), (10, 0.2), (10, 0.3), (5, 0.1), (20, 0.1)):
            lines.append(f"{short if first else ''} & {k} & {mg:.1f} & {m_(meth,k,mg,'fpr'):.3f} & "
                         f"{m_(meth,k,mg,'recall'):.3f} & {m_(meth,k,mg,'precision'):.3f} & {m_(meth,k,mg,'f1'):.3f} \\\\")
            first = False
        lines[-1] += " \\midrule" if meth == "Gabor-Gaussian" else ""
    _w("tab_threshold", r"""\begin{table}[t]
\centering
\caption{Decisions with the label-free LOO threshold (\ref{eq:tau}); mean over five categories and three draws. FPR: false alarms on good test images.}
\label{tab:threshold}
\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
Model & $k$ & $m$ & FPR & Recall & Prec. & F1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tex_ablation(ab):
    variants = list(dict.fromkeys(r["variant"] for r in ab))
    lines = []
    for v in variants:
        rs = [r for r in ab if r["variant"] == v]
        lines.append(f"{v} & {rs[0]['dim']} & {np.mean([r['img_auroc'] for r in rs]):.3f} & "
                     f"{np.mean([r['px_auroc'] for r in rs]):.3f} & {np.mean([r['aupro'] for r in rs]):.3f} \\\\")
    _w("tab_ablation", r"""\begin{table}[t]
\centering
\caption{Ablation of the station descriptor ($k=20$, mean over five categories).}
\label{tab:ablation}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
Variant & $D$ & Img. & Pix. & AUPRO \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tex_robust(rb):
    perts = list(dict.fromkeys(r["perturb"] for r in rb))
    lines = []
    for m, short in zip(METHODS, ("GG", "PC", "PD")):
        base = np.mean([r["img_auroc"] for r in rb if r["method"] == m and r["perturb"] == "none"])
        cells = [f"{base:.3f}"]
        for p in perts[1:]:
            v = np.mean([r["img_auroc"] for r in rb if r["method"] == m and r["perturb"] == p])
            cells.append(f"{v:.3f} ({v - base:+.3f})")
        lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
    hdr = " & ".join(p.replace("%", "\\%").replace("s=", "$\\sigma$=") for p in perts)
    _w("tab_robust", r"""\begin{table*}[t]
\centering
\caption{Robustness of image AUROC to imaging changes applied to the test frames only ($k=20$, mean over five categories; change w.r.t.\ clean in parentheses).}
\label{tab:robust}
\footnotesize
\begin{tabular}{@{}l""" + "c" * len(perts) + r"""@{}}
\toprule
Model & """ + hdr + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def fig_gate(cas):
    rows = cas["rows"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for b, ls in ((1.5, "-"), (2.0, "--"), (float("inf"), ":")):
        pts = []
        for a in sorted({r["alpha"] for r in rows if r["policy"] == "cascade"}):
            rs = [r for r in rows if r["policy"] == "cascade" and r["alpha"] == a and r["beta"] == b]
            pts.append((np.mean([r["offload"] for r in rs]), np.mean([r["f1"] for r in rs]), a))
        x, y, a = zip(*pts)
        ax.plot(x, y, ls, color=COL["PatchCore-lite"], lw=1.4, marker="o", ms=3,
                label=f"cascade, β={'∞' if b == float('inf') else b}")
    for pol, mk, lab in (("station-only", "D", "station only"), ("edge-only", "s", "edge only")):
        rs = [r for r in rows if r["policy"] == pol]
        ax.plot(np.mean([r["offload"] for r in rs]), np.mean([r["f1"] for r in rs]), mk, ms=6,
                color=COL["Gabor-Gaussian"] if pol == "station-only" else INK, label=lab)
    rs = [r for r in rows if r["policy"] == "cascade" and r["alpha"] == 0.6 and r["beta"] == 1.5]
    ax.annotate("default\n(α=0.6, β=1.5)", (np.mean([r["offload"] for r in rs]), np.mean([r["f1"] for r in rs])),
                (0.08, 0.735), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel("fraction of frames offloaded"); ax.set_ylabel("F1 (mean of 5 textures)")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=6.3, loc="lower right")
    fig.savefig(os.path.join(F, "fig_gate.png")); plt.close(fig)


def tex_gate_cat(cas):
    rows = cas["rows"]
    lines = []
    for cat in CATS + ["mean"]:
        cells = []
        for pol, a, b in (("station-only", None, None), ("cascade", 0.6, 1.5), ("edge-only", None, None)):
            rs = [r for r in rows if r["policy"] == pol and (cat == "mean" or r["cat"] == cat)
                  and (a is None or (r["alpha"] == a and r["beta"] == b))]
            cells += [f"{np.mean([r['f1'] for r in rs]):.3f}", f"{np.mean([r['fpr'] for r in rs]):.3f}"]
            if pol == "cascade":
                cells.append(f"{np.mean([r['offload'] for r in rs]):.2f}")
        name = "\\textit{Mean}" if cat == "mean" else cat
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\" + (" \\midrule" if cat == "wood" else ""))
    _w("tab_gate", r"""\begin{table}[t]
\centering
\caption{Per-fabric decisions of the three policies ($k=10$, $m=0.1$, mean of three draws). Cascade uses the a-priori band $\alpha=0.6,\beta=1.5$; ``off.'' is the offloaded fraction.}
\label{tab:gate}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{2}{c}{Station only} & \multicolumn{3}{c}{Cascade} & \multicolumn{2}{c}{Edge only} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}
Category & F1 & FPR & F1 & FPR & off. & F1 & FPR \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


LIVE_KEYS = ("offload", "lat_p50", "lat_p95", "deadline_miss", "uplink_kbps_per_station", "edge_util",
             "mean_batch", "f1", "completed")


def load_live():
    """Merge final live runs and aggregate repetitions -> rows with <key> (mean) and <key>_sd."""
    rows = []
    for n in ("iot_live_wifi", "iot_live_constrained"):
        d = load(n)
        if d:
            rows += d["rows"]
    if not rows:
        return None
    g = {}
    for r in rows:
        g.setdefault((r["link"], r["policy"], r["stations"]), []).append(r)
    out = []
    for (link, pol, n), rs in g.items():
        a = dict(link=link, policy=pol, stations=n, fps=rs[0]["fps"], reps=len(rs))
        for k in LIVE_KEYS:
            v = np.array([r[k] for r in rs], float)
            a[k], a[k + "_sd"] = float(v.mean()), float(v.std())
        out.append(a)
    return {"rows": out}


def fig_live(live, eb=None):
    rows = live["rows"]
    pols = [("station-only", COL["Gabor-Gaussian"], "D"), ("cascade", COL["PatchCore-lite"], "o"),
            ("edge-only", INK2, "s")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for j, link in enumerate(("wifi", "constrained")):
        ax = axs[j]
        for pol, c, mk in pols:
            rs = sorted([r for r in rows if r["link"] == link and r["policy"] == pol], key=lambda r: r["stations"])
            if not rs:
                continue
            ax.errorbar([r["stations"] for r in rs], [r["lat_p95"] for r in rs],
                        yerr=[r["lat_p95_sd"] for r in rs], color=c, marker=mk, ms=3.5,
                        lw=1.5, capsize=2, elinewidth=0.8, label=pol)
        ax.axhline(200, color=INK2, lw=0.7, ls="--")
        ax.text(1.05, 210, "frame period (200 ms)", fontsize=6, color=INK2)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 4, 8, 16, 32]); ax.set_xticklabels(["1", "4", "8", "16", "32"]); ax.minorticks_off()
        ax.set_xlabel("stations N"); ax.set_title(f"p95 decision latency – {link}", fontsize=8, color=INK)
    axs[0].set_ylabel("ms")
    ax = axs[2]
    for pol, c, mk in pols[1:]:
        rs = sorted([r for r in rows if r["link"] == "wifi" and r["policy"] == pol], key=lambda r: r["stations"])
        ax.errorbar([r["stations"] for r in rs], [r["edge_util"] for r in rs], yerr=[r["edge_util_sd"] for r in rs],
                    color=c, marker=mk, ms=3.5, lw=1.5, capsize=2, elinewidth=0.8, label=pol + " (measured)")
        if eb:
            mu = max(e["mu_fps"] for e in eb["edge"])
            ax.plot([r["stations"] for r in rs], [min(r["stations"] * r["fps"] * r["offload"] / mu, 1.2) for r in rs],
                    color=c, lw=0.9, ls=":", label=pol + " (model, Eq. 6)")
    ax.axhline(1.0, color=INK2, lw=0.7, ls="--")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 8, 16, 32]); ax.set_xticklabels(["1", "4", "8", "16", "32"])
    ax.minorticks_off(); ax.set_xlabel("stations N"); ax.set_title("edge utilisation – wifi", fontsize=8, color=INK)
    ax.legend(fontsize=5.5, loc="upper left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.36, 1.08))
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_live.png")); plt.close(fig)


def tex_live(live):
    lines = []
    for link in ("wifi", "constrained"):
        for pol in ("station-only", "cascade", "edge-only"):
            for r in sorted([r for r in live["rows"] if r["link"] == link and r["policy"] == pol],
                            key=lambda r: r["stations"]):
                if r["stations"] not in (1, 8, 32):
                    continue
                sd = lambda k, f, s=1: f" $\\pm${s * r[k + '_sd']:{f}}" if r.get("reps", 1) > 1 else ""
                lines.append(f"{link} & {pol} & {r['stations']} & {r['offload']:.2f} & {r['lat_p50']:.0f} & "
                             f"{r['lat_p95']:.0f}{sd('lat_p95', '.0f')} & {100 * r['deadline_miss']:.1f}{sd('deadline_miss', '.1f', 100)} & "
                             f"{r['uplink_kbps_per_station']:.0f} & {r['edge_util']:.2f}{sd('edge_util', '.2f')} & {r['f1']:.3f} \\\\")
            lines[-1] += " \\midrule" if not (link == "constrained" and pol == "edge-only") else ""
    _w("tab_live", r"""\begin{table*}[t]
\centering
\caption{Live deployment (Mosquitto broker, one edge server, $N$ single-core stations at 5 frames/s, $k=10$). Latency is capture-to-decision at the station; miss: decisions later than one frame period or never received; uplink per station; edge utilisation over the 12-s capture window. Wi-Fi: mean $\pm$ std of three repetitions; constrained: one run.}
\label{tab:live}
\footnotesize
\begin{tabular}{@{}llcccccccc@{}}
\toprule
Link & Policy & $N$ & Offload & p50 [ms] & p95 [ms] & Miss [\%] & Uplink [kbit/s] & Edge util. & F1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tex_resources(lat, eb, prof=None):
    g = {(r["method"], r["threads"]): r for r in lat}
    st = eb["station"]
    mu = {e["batch"]: e for e in eb["edge"]}
    size = {}
    if prof:
        size["Gabor-Gaussian"] = np.mean([r["gabor_bytes"] for r in prof["sizes"]]) / 1e3
        size["PatchCore-lite"] = np.mean([r["patchcore_bytes"] for r in prof["sizes"]]) / 1e3
    lines = []
    for m, short in zip(METHODS, ("GG", "PC", "PD")):
        prof_kb = f"{size[m]:.1f}" if m in size else f"{1e3 * g[(m, 8)]['model_state_MB']:.0f}$^\\dagger$"
        lines.append(f"{short} & {g[(m, 8)]['fit_s']:.2f} & {g[(m, 8)]['ms_median']:.1f} & {g[(m, 2)]['ms_median']:.1f} & "
                     f"{prof_kb} & {g[(m, 8)]['backbone_MB']:.1f} \\\\")
    _w("tab_resources", r"""\begin{table}[t]
\centering
\caption{Computational resources per $256\times256$ frame ($k=20$; median of 30 runs, single process). Profile: serialised model per fabric; $^\dagger$in-memory statistics (not serialised in the application).}
\label{tab:resources}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
& Fit [s] & \multicolumn{2}{c}{Inference [ms]} & Profile & Backbone \\
\cmidrule(lr){3-4}
Model & & 8 thr. & 2 thr. & [kB] & [MB] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\begin{flushleft}\footnotesize
Station (1 thread, $k=10$): """ + f"{st['station_ms_median']:.1f} ms median, {st['station_ms_p95']:.1f} ms p95. " +
"Edge throughput $\\mu(b)$ [frames/s]: " + ", ".join(f"$b{{=}}{b}$: {mu[b]['mu_fps']:.0f}" for b in sorted(mu)) + r""".
\end{flushleft}
\end{table}
""")


def fig_architecture():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.3, color=INK2,
                linespacing=1.3)

    # stations
    for i, y in enumerate((26.5, 15.5, 2.5)):
        lab = "Station " + ("1", "2", "N")[i]
        box(1, y, 27, 8.5, lab, "USB camera, 1 CPU core\nGabor–Gaussian model + gate")
    ax.text(14.5, 13.0, "⋮", ha="center", va="center", fontsize=9, color=INK2)
    box(37, 11, 22, 14, "MQTT broker", "Mosquitto, plant LAN/Wi-Fi\nretained profiles\noffload / result / verdict", "#e8eef8")
    box(70, 20, 29, 13, "Edge server (CPU)", "PatchCore-lite (ResNet-18)\nmicro-batching b ≤ 8\nper-fabric profiles")
    box(70, 3, 29, 13, "Profile registry", "learn from k defect-free frames\nfit both models + LOO τ\npublish station profile (retained)")
    arr = dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0)
    arr2 = dict(arrowstyle="<|-", color="#2a78d6", lw=0.9, shrinkA=0, shrinkB=0)
    for y in (30.5, 19.5, 6.5):
        ax.annotate("", (36.5, 18 + (y - 19) * 0.25), (28.6, y), arrowprops=dict(arrowstyle="<|-|>", color=INK2, lw=0.9))
    ax.text(32.5, 33.6, "uncertain frames ↑\nresults, profiles ↓", ha="center", fontsize=5.8, color=INK2)
    ax.annotate("", (69.5, 27), (59.5, 21), arrowprops=arr)
    ax.annotate("", (59.5, 19), (69.5, 25), arrowprops=dict(arr, color="#eb6834"))
    ax.text(64.5, 27.6, "offloads", ha="center", fontsize=5.8, color=INK2)
    ax.text(64.5, 16.8, "results", ha="center", fontsize=5.8, color="#eb6834")
    ax.annotate("", (59.5, 14), (69.5, 9.5), arrowprops=dict(arr, color="#2a78d6"))
    ax.text(64.5, 8.2, "profiles", ha="center", fontsize=5.8, color="#2a78d6")
    fig.savefig(os.path.join(F, "fig_architecture.png")); plt.close(fig)


if __name__ == "__main__":
    fs = load("fewshot")
    if fs: fig_fewshot(fs)
    th = load("threshold")
    if th: tab_threshold(th)
    for n in ("robust", "ablation", "latency"):
        v = load(n)
        if v: tables[n] = v
    fig_architecture()
    if fs: tex_k10(fs)
    if th: tex_threshold(th)
    if tables.get("ablation"): tex_ablation(tables["ablation"])
    if tables.get("robust"): tex_robust(tables["robust"])
    cas = load("iot_cascade")
    if cas: fig_gate(cas); tex_gate_cat(cas)
    live, eb = load_live(), load("iot_edgebench")
    if live: fig_live(live, eb); tex_live(live)
    if eb and tables.get("latency"): tex_resources(tables["latency"], eb, load("iot_profile"))
    if "noqual" not in sys.argv:
        fig_qualitative()
    json.dump(tables, open(os.path.join(R, "tables.json"), "w"), indent=1)
    print("figures:", os.listdir(F))
