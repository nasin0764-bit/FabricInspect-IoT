"""Smoke tests on synthetic textures (no dataset needed):  python -m pytest tests"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fabricinspect import detectors as D
from fabricinspect.engine import Inspector


def weave(seed, defect=False):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:256, 0:256]
    g = 128 + 50 * np.sin(x / 3.0 + rng.uniform(0, 6)) * np.sin(y / 3.0 + rng.uniform(0, 6))
    g = g + rng.normal(0, 6, g.shape)
    if defect:
        g[120:136, 60:200] = 40  # broken-pick-like dark band
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _check(kind):
    det = D.DETECTORS[kind]().fit([weave(i) for i in range(6)])
    assert det.image_threshold > 0 and len(det.loo_scores) == 6
    good, bad = det.predict(weave(100)), det.predict(weave(101, defect=True))
    assert bad["score"] > good["score"]
    assert bad["defect"] and not good["defect"]
    return det


def test_gabor():
    _check("gabor")


def test_patchcore_and_roundtrip(tmp_path):
    det = _check("patchcore")
    p = str(tmp_path / "m.npz")
    D.save(det, p)
    det2 = D.load(p)
    img = weave(7, defect=True)
    assert np.allclose(det.predict(img)["map"], det2.predict(img)["map"])
    assert det2.image_threshold == det.image_threshold


def test_iot_pipeline():
    from fabricinspect.iot.broker import start_in_thread
    from fabricinspect.iot.edge import EdgeServer
    from fabricinspect.iot.registry import learn_both, publish_profile
    from fabricinspect.iot.station import Station
    port = 18899
    start_in_thread(port=port)
    g, pc = learn_both([weave(i) for i in range(6)])
    assert publish_profile("syn", g, port=port) < 20_000
    srv = EdgeServer(port=port, threads=2)
    srv.add_model("syn", pc)
    frames = [weave(200 + i, defect=i % 2 == 1) for i in range(6)]
    for policy in ("local", "edge"):
        st = Station(f"t{policy}", "syn", port=port, policy=policy)
        st.wait_profile(10)
        recs = st.run(frames, fps=20)
        st.close()
        assert all(r["done"] for r in recs)
        assert [r["defect"] for r in recs] == [i % 2 == 1 for i in range(6)]
        assert all(r["tier"] == ("edge" if policy == "edge" else "station") for r in recs)


def test_inspector_logs(tmp_path, monkeypatch):
    import fabricinspect.engine as E
    monkeypatch.setattr(E, "PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(E, "LOG_DIR", str(tmp_path / "logs"))
    ins = Inspector("gabor")
    ins.learn([weave(i) for i in range(4)], "synthetic")
    r = ins.inspect(weave(55, defect=True))
    assert r["defect"] and r["overlay"].shape == (256, 256, 3)
    assert os.listdir(tmp_path / "logs")
