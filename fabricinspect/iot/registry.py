"""Profile registry: learn a fabric once, distribute it to every station.

The Gabor-Gaussian station profile is published as a *retained* MQTT message,
so stations that start (or reconnect) later receive it immediately. The
PatchCore-lite profile stays on the edge server.

    python -m fabricinspect.iot.registry --name denim --images good_frames/
"""
import argparse
import os

import paho.mqtt.client as mqtt

from . import protocol as P
from .. import detectors as D
from ..engine import PROFILE_DIR, center_square, read_images


def learn_both(frames, margin=0.10, roi=1.0):
    sq = [center_square(f, roi) for f in frames]
    return D.GaborGaussian().fit(sq, margin=margin), D.PatchCoreLite().fit(sq, margin=margin)


def publish_profile(fabric, gabor, host="127.0.0.1", port=1883):
    b = P.profile_bytes(gabor)
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"registry-{fabric}")
    c.connect(host, port)
    c.loop_start()
    c.publish(P.t_profile(fabric, "gabor"), b, qos=1, retain=True).wait_for_publish(10)
    c.loop_stop(); c.disconnect()
    return len(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    a = ap.parse_args()
    g, pc = learn_both([im for _, im in read_images(a.images)])
    os.makedirs(PROFILE_DIR, exist_ok=True)
    D.save(pc, os.path.join(PROFILE_DIR, a.name + ".npz"))   # for the edge server
    n = publish_profile(a.name, g, a.host, a.port)
    print(f"published station profile '{a.name}' ({n} bytes); edge profile saved to {PROFILE_DIR}")
