"""FabricInspect-IoT message protocol (MQTT topics and payloads).

Topic                               QoS  Retain  Payload
fi/profile/<fabric>/gabor           1    yes     station profile (.npz bytes, Gabor-Gaussian)
fi/st/<station>/offload             1    no      OFFLOAD frame: 4-byte header length, JSON header, JPEG
fi/st/<station>/result              1    no      JSON {seq, score, defect, threshold, server_ms, batch}
fi/st/<station>/verdict             0    no      JSON {seq, fabric, defect, score, tier, latency_ms}
fi/edge/status                      0    yes     JSON {queue, served, util}
"""
import io
import json
import struct

import cv2
import numpy as np

from .. import detectors as D

ROOT = "fi"


def t_profile(fabric, kind="gabor"):
    return f"{ROOT}/profile/{fabric}/{kind}"


def t_offload(station="+"):
    return f"{ROOT}/st/{station}/offload"


def t_result(station):
    return f"{ROOT}/st/{station}/result"


def t_verdict(station="+"):
    return f"{ROOT}/st/{station}/verdict"


T_EDGE_STATUS = f"{ROOT}/edge/status"


def station_of(topic: str) -> str:
    return topic.split("/")[2]


def pack_offload(header: dict, img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    h = json.dumps(header).encode()
    return struct.pack("!I", len(h)) + h + jpg.tobytes()


def unpack_offload(payload: bytes):
    n = struct.unpack_from("!I", payload)[0]
    header = json.loads(payload[4:4 + n])
    img = cv2.imdecode(np.frombuffer(payload[4 + n:], np.uint8), cv2.IMREAD_COLOR)
    return header, img


def profile_bytes(det: D.Detector) -> bytes:
    buf = io.BytesIO()
    D.save(det, buf)
    return buf.getvalue()


def profile_from_bytes(b: bytes) -> D.Detector:
    return D.load(io.BytesIO(b))
