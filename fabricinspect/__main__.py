"""FabricInspect command line.

    python -m fabricinspect gui
    python -m fabricinspect learn   --name denim_blue --images ./good_frames [--detector gabor]
    python -m fabricinspect inspect --name denim_blue --images ./to_check
"""
import argparse
import json

from .engine import Inspector, list_profiles, read_images


def main():
    ap = argparse.ArgumentParser(prog="fabricinspect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gui")
    sub.add_parser("profiles")
    lp = sub.add_parser("learn")
    lp.add_argument("--name", required=True)
    lp.add_argument("--images", required=True, help="folder of defect-free images")
    lp.add_argument("--detector", default="patchcore", choices=["patchcore", "gabor"])
    lp.add_argument("--roi", type=float, default=1.0)
    lp.add_argument("--margin", type=float, default=0.10)
    ip = sub.add_parser("inspect")
    ip.add_argument("--name", required=True)
    ip.add_argument("--images", required=True)
    ip.add_argument("--roi", type=float, default=1.0)
    ip.add_argument("--webhook", default=None)
    # IoT tiers
    bp = sub.add_parser("broker", help="bundled MQTT broker (use Mosquitto in production)")
    bp.add_argument("--port", type=int, default=1883)
    ep = sub.add_parser("edge", help="edge server (PatchCore-lite verification)")
    ep.add_argument("--host", default="127.0.0.1")
    ep.add_argument("--port", type=int, default=1883)
    pp = sub.add_parser("publish", help="learn a fabric and distribute its station profile")
    pp.add_argument("--name", required=True)
    pp.add_argument("--images", required=True)
    pp.add_argument("--host", default="127.0.0.1")
    pp.add_argument("--port", type=int, default=1883)
    sp = sub.add_parser("station", help="station node at a seaming machine")
    sp.add_argument("--name", required=True)
    sp.add_argument("--fabric", required=True)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=1883)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--fps", type=float, default=5.0)
    sp.add_argument("--policy", default="cascade", choices=["cascade", "local", "edge"])
    sp.add_argument("--alpha", type=float, default=0.6)
    sp.add_argument("--beta", type=float, default=1.5)
    a = ap.parse_args()

    if a.cmd == "broker":
        import asyncio
        from .iot.broker import Broker
        asyncio.run(Broker().serve("0.0.0.0", a.port))
    elif a.cmd == "edge":
        import time
        from .engine import PROFILE_DIR
        from .iot.edge import EdgeServer
        srv = EdgeServer(a.host, a.port)
        srv.load_profiles(PROFILE_DIR)
        print("edge server ready; fabrics:", sorted(srv.models))
        while True:
            time.sleep(10)
            print(f"served={srv.served} util={srv.utilisation():.2f} queue={srv.q.qsize()}")
    elif a.cmd == "publish":
        import os
        from . import detectors as D
        from .engine import PROFILE_DIR
        from .iot.registry import learn_both, publish_profile
        g, pc = learn_both([im for _, im in read_images(a.images)])
        os.makedirs(PROFILE_DIR, exist_ok=True)
        D.save(pc, os.path.join(PROFILE_DIR, a.name + ".npz"))
        n = publish_profile(a.name, g, a.host, a.port)
        print(f"station profile '{a.name}' published ({n} B); edge profile in {PROFILE_DIR}")
    elif a.cmd == "station":
        from .iot.station import Station
        st = Station(a.name, a.fabric, a.host, a.port, a.policy, a.alpha, a.beta)
        st.wait_profile()
        st.run_camera(a.camera, a.fps, show=True)
    elif a.cmd == "gui":
        from .gui import run
        run()
    elif a.cmd == "profiles":
        print("\n".join(list_profiles()))
    elif a.cmd == "learn":
        ins = Inspector(a.detector, a.roi, a.margin)
        info = ins.learn([im for _, im in read_images(a.images)], a.name)
        print(json.dumps(info, indent=1))
    elif a.cmd == "inspect":
        ins = Inspector(roi=a.roi, webhook=a.webhook)
        ins.load_profile(a.name)
        for p, im in read_images(a.images):
            r = ins.inspect(im, source=p)
            print(f"{'DEFECT' if r['defect'] else 'ok    '} score={r['score']:.3f} "
                  f"thr={r['threshold']:.3f} area={r['defect_area_frac']:.4f} "
                  f"{r['ms']:.0f}ms  {p}")


if __name__ == "__main__":
    main()
