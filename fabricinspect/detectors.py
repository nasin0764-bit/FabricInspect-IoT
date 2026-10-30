"""Few-shot, label-free fabric defect detectors used by FabricInspect.

All detectors learn only from a handful of defect-free images of the fabric
currently on the machine and share one interface:

    det.fit(images)              # list of BGR uint8 images, defect-free
    det.anomaly_map(image)       # float32 [size, size] map, higher = worse
    det.predict(image)           # dict(score, defect, defect_area_frac, map)

Thresholds are calibrated without defect labels by leave-one-out (LOO) over
the defect-free shots: each shot is scored by a model fitted on the others.

Detectors
---------
GaborGaussian   classical: pooled Gabor energies + Lab colour statistics,
                one shrinkage Gaussian shared by all positions.
PatchCoreLite   ImageNet ResNet-18 layer2+3 patch features, random-subsampled
                memory bank, nearest-neighbour distance (after Roth et al. 2022).
PaDiM           ResNet-18 layer1-3, 100 random channels, one Gaussian per
                position (after Defard et al. 2021). No LOO calibration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

SIZE = 256


def _finish(m: np.ndarray, size: int = SIZE, sigma: float = 4.0) -> np.ndarray:
    m = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), sigma)


class Detector:
    name = "base"
    supports_loo = True

    def __init__(self):
        self.image_threshold = None
        self.loo_scores = None

    # subclasses implement these three
    def embed(self, imgs: list) -> list: ...
    def fit_embeds(self, embs: list): ...
    def map_from_embed(self, emb) -> np.ndarray: ...

    def fit(self, imgs: list, calibrate: bool = True, margin: float = 0.10) -> "Detector":
        embs = self.embed(imgs)
        if calibrate and self.supports_loo and len(embs) >= 2:
            self.calibrate(embs, margin)
        self.fit_embeds(embs)
        return self

    def calibrate(self, embs, margin=0.10):
        """Threshold = (1 + margin) x the worst leave-one-out score of the shots."""
        scores = []
        for i in range(len(embs)):
            self.fit_embeds(embs[:i] + embs[i + 1:])
            scores.append(float(self.map_from_embed(embs[i]).max()))
        self.loo_scores = np.array(scores)
        self.image_threshold = float(self.loo_scores.max() * (1 + margin))

    def anomaly_map(self, img: np.ndarray) -> np.ndarray:
        return self.map_from_embed(self.embed([img])[0])

    def predict(self, img: np.ndarray) -> dict:
        m = self.anomaly_map(img)
        s = float(m.max())
        thr = self.image_threshold
        return {
            "score": s,
            "defect": None if thr is None else bool(s > thr),
            "defect_area_frac": None if thr is None else float((m > thr).mean()),
            "map": m,
        }


# --------------------------------------------------------------------------
# Classical detector
# --------------------------------------------------------------------------
@dataclass
class GGConfig:
    wavelengths: tuple = (4.0, 8.0, 16.0)  # Gabor wavelengths, px at 256x256
    n_orient: int = 4
    pool: int = 15                         # local pooling window, px
    stride: int = 4                        # feature-grid stride, px
    use_intensity: bool = True
    use_color: bool = True                 # Lab a*, b* local mean/std
    use_gabor: bool = True
    use_gabor_std: bool = True
    shrinkage: float = 0.01
    max_vectors: int = 200_000
    seed: int = 0


def _box(x, w):
    return cv2.blur(x, (w, w), borderType=cv2.BORDER_REFLECT)


def _mean_std(x, w):
    m = _box(x, w)
    return m, np.sqrt(np.maximum(_box(x * x, w) - m * m, 0))


class GaborGaussian(Detector):
    name = "Gabor-Gaussian"

    def __init__(self, cfg: GGConfig | None = None):
        super().__init__()
        self.cfg = cfg or GGConfig()
        self.kernels = []
        for lam in self.cfg.wavelengths:
            sig = 0.56 * lam
            k = int(2 * round(3 * sig) + 1)
            for i in range(self.cfg.n_orient):
                th = np.pi * i / self.cfg.n_orient
                re = cv2.getGaborKernel((k, k), sig, th, lam, 0.5, 0, cv2.CV_32F)
                im = cv2.getGaborKernel((k, k), sig, th, lam, 0.5, np.pi / 2, cv2.CV_32F)
                re -= re.mean()
                self.kernels.append((re / np.abs(re).sum(), im / np.abs(im).sum()))
        self._spectra = None

    def _gabor_energies(self, g):
        """|g (x) (even + i*odd)| for all filters via one FFT of the reflect-padded image.

        Equivalent to cv2.filter2D(..., BORDER_REFLECT) on both parts (the pad is
        at least the largest kernel radius), but ~10x cheaper on one core.
        """
        from scipy import fft
        if self._spectra is None:
            pad = max(k[0].shape[0] for k in self.kernels) // 2 + 1
            n = fft.next_fast_len(SIZE + 2 * pad)
            hs = []
            for re, im in self.kernels:
                r = re.shape[0] // 2
                h = np.zeros((n, n), np.complex64)
                idx = np.arange(-r, r + 1) % n
                h[np.ix_(idx, idx)] = re + 1j * im
                # correlation: conj of each real kernel's spectrum, combined
                hs.append(np.conj(fft.fft2(h.real)) + 1j * np.conj(fft.fft2(h.imag)))
            self._spectra = (pad, n, np.stack(hs).astype(np.complex64))
        pad, n, H = self._spectra
        gp = np.pad(g, ((pad, n - SIZE - pad),) * 2, mode="symmetric")
        out = fft.ifft2(fft.fft2(gp.astype(np.complex64))[None] * H, axes=(-2, -1), workers=1)
        return np.abs(out[:, pad:pad + SIZE, pad:pad + SIZE]).astype(np.float32)

    def _grid(self, img):
        c, w = self.cfg, self.cfg.pool
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            g, ab = img.astype(np.float32) / 255, ()
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
            g, ab = lab[..., 0], (lab[..., 1], lab[..., 2])
        maps = []
        if c.use_intensity:
            maps += _mean_std(g, w)
        if c.use_color:
            for ch in ab:
                maps += _mean_std(ch, w)
        if c.use_gabor:
            for e in self._gabor_energies(g):
                m, sd = _mean_std(e, w)
                maps += [m, sd] if c.use_gabor_std else [m]
        s = c.stride
        return np.stack(maps, -1)[s // 2::s, s // 2::s]

    def embed(self, imgs):
        return [self._grid(i) for i in imgs]

    def fit_embeds(self, embs):
        x = np.concatenate([e.reshape(-1, e.shape[-1]) for e in embs]).astype(np.float64)
        if len(x) > self.cfg.max_vectors:
            rng = np.random.default_rng(self.cfg.seed)
            x = x[rng.choice(len(x), self.cfg.max_vectors, replace=False)]
        self.mean = x.mean(0)
        cov = np.cov(x - self.mean, rowvar=False)
        cov += self.cfg.shrinkage * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.prec = np.linalg.inv(cov)

    def map_from_embed(self, e):
        d = e.reshape(-1, e.shape[-1]).astype(np.float64) - self.mean
        m = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec, d), 0))
        return _finish(m.reshape(e.shape[:2]))

    def state(self):
        return {"mean": self.mean, "prec": self.prec}, {"cfg": asdict(self.cfg)}

    def set_state(self, arrays, meta):
        cfg = dict(meta["cfg"]); cfg["wavelengths"] = tuple(cfg["wavelengths"])
        self.__init__(GGConfig(**cfg))
        self.mean, self.prec = arrays["mean"], arrays["prec"]


# --------------------------------------------------------------------------
# Deep-feature detectors (torch is imported lazily)
# --------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_BACKBONE = None


def _resnet18():
    global _BACKBONE
    if _BACKBONE is None:
        import torchvision
        _BACKBONE = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    return _BACKBONE


def _layers(imgs, bs=16):
    """Yield (l1, l2, l3) ResNet-18 feature batches for BGR images."""
    import torch
    net = _resnet18()
    for i in range(0, len(imgs), bs):
        x = np.stack([(cv2.cvtColor(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - _MEAN) / _STD
                      for im in imgs[i:i + bs]])
        with torch.no_grad():
            x = torch.from_numpy(x).permute(0, 3, 1, 2)
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            l1 = net.layer1(x); l2 = net.layer2(l1); l3 = net.layer3(l2)
        yield l1, l2, l3


class PatchCoreLite(Detector):
    name = "PatchCore-lite"

    def __init__(self, ratio: float = 0.1, min_bank: int = 1000, seed: int = 0):
        super().__init__()
        self.ratio, self.min_bank, self.seed = ratio, min_bank, seed

    def embed(self, imgs):
        import torch
        import torch.nn.functional as F
        out = []
        for _, l2, l3 in _layers(imgs):
            l2 = F.avg_pool2d(l2, 3, 1, 1)
            l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:],
                               mode="bilinear", align_corners=False)
            e = torch.cat([l2, l3], 1).permute(0, 2, 3, 1)  # N,32,32,384
            out += list(e)
        return out

    def fit_embeds(self, embs):
        import torch
        bank = torch.cat([e.reshape(-1, e.shape[-1]) for e in embs])
        k = min(len(bank), max(int(len(bank) * self.ratio), self.min_bank))
        g = torch.Generator().manual_seed(self.seed)
        self.bank = bank[torch.randperm(len(bank), generator=g)[:k]].contiguous()

    def map_from_embed(self, e):
        import torch
        q = e.reshape(-1, e.shape[-1])
        d = torch.cdist(q, self.bank).min(1).values.numpy()
        return _finish(d.reshape(e.shape[:2]))

    def state(self):
        return {"bank": self.bank.numpy()}, {"ratio": self.ratio, "min_bank": self.min_bank,
                                             "seed": self.seed}

    def set_state(self, arrays, meta):
        import torch
        self.__init__(**meta)
        self.bank = torch.from_numpy(arrays["bank"])


class PaDiM(Detector):
    name = "PaDiM"
    supports_loo = False  # per-position statistics make LOO prohibitively large

    def __init__(self, d: int = 100, seed: int = 0):
        super().__init__()
        import torch
        self.d = d
        self.idx = torch.randperm(448, generator=torch.Generator().manual_seed(seed))[:d]

    def embed(self, imgs):
        import torch
        import torch.nn.functional as F
        out = []
        for l1, l2, l3 in _layers(imgs):
            s = l1.shape[-2:]
            e = torch.cat([l1, F.interpolate(l2, size=s, mode="nearest"),
                           F.interpolate(l3, size=s, mode="nearest")], 1)[:, self.idx]
            out += list(e.numpy())
        return out  # each d,64,64

    def fit_embeds(self, embs):
        e = np.stack(embs)
        n, d, h, w = e.shape
        e = e.reshape(n, d, h * w).transpose(2, 0, 1)
        self.mu = e.mean(1)
        import torch
        # batched covariance / inverse in torch: ~50x faster than numpy's per-matrix loops here
        x = torch.from_numpy(np.ascontiguousarray(e - self.mu[:, None], dtype=np.float32))
        cov = torch.bmm(x.transpose(1, 2), x) / max(n - 1, 1) + 0.01 * torch.eye(d)[None]
        self.prec = torch.linalg.inv(cov).numpy()
        self.hw = (h, w)

    def map_from_embed(self, e):
        x = (e.reshape(self.d, -1).T - self.mu).astype(np.float32)
        m = np.sqrt(np.maximum((np.matmul(self.prec, x[:, :, None])[..., 0] * x).sum(1), 0))
        return _finish(m.reshape(self.hw))


DETECTORS = {"gabor": GaborGaussian, "patchcore": PatchCoreLite}


def save(det: Detector, path: str):
    arrays, meta = det.state()
    meta = {"kind": next(k for k, v in DETECTORS.items() if isinstance(det, v)),
            "image_threshold": det.image_threshold,
            **meta}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load(path: str) -> Detector:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    cls = DETECTORS[meta.pop("kind")]
    img_thr = meta.pop("image_threshold")
    det = cls.__new__(cls)
    det.set_state({k: z[k] for k in z.files if k != "meta"}, meta)  # re-runs __init__
    det.image_threshold = img_thr
    return det
