"""Minimal MQTT 3.1.1 broker (asyncio) bundled for self-contained experiments.

Supports CONNECT, PUBLISH (QoS 0/1, retain), PUBACK, SUBSCRIBE/SUBACK with
'+'/'#' wildcards, UNSUBSCRIBE, PINGREQ and DISCONNECT. Messages are delivered
at min(publish QoS, subscription QoS). No persistence, auth or QoS 2: in a
production deployment, point the stations at Mosquitto / EMQX instead - the
clients are standard paho-mqtt and need no change.

    python -m fabricinspect.iot.broker --port 1883
"""
import argparse
import asyncio
import struct


def _match(filt: str, topic: str) -> bool:
    f, t = filt.split("/"), topic.split("/")
    for i, p in enumerate(f):
        if p == "#":
            return True
        if i >= len(t) or (p != "+" and p != t[i]):
            return False
    return len(f) == len(t)


def _enc_len(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n % 128; n //= 128
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _str(b: bytes, i: int):
    n = struct.unpack_from("!H", b, i)[0]
    return b[i + 2:i + 2 + n], i + 2 + n


class Session:
    def __init__(self, broker, reader, writer):
        self.b, self.r, self.w = broker, reader, writer
        self.subs = {}          # filter -> qos
        self.pid = 0

    async def read_packet(self):
        h = await self.r.readexactly(1)
        mult, n = 1, 0
        while True:
            d = (await self.r.readexactly(1))[0]
            n += (d & 127) * mult; mult *= 128
            if not d & 128:
                break
        return h[0], await self.r.readexactly(n)

    def send(self, header: int, body: bytes):
        self.w.write(bytes([header]) + _enc_len(len(body)) + body)

    def deliver(self, topic: bytes, payload: bytes, qos: int, retain=False):
        body = struct.pack("!H", len(topic)) + topic
        if qos:
            self.pid = self.pid % 65535 + 1
            body += struct.pack("!H", self.pid)
        self.send(0x30 | (qos << 1) | int(retain), body + payload)

    async def run(self):
        try:
            while True:
                h, body = await self.read_packet()
                t = h >> 4
                if t == 1:                                    # CONNECT
                    self.send(0x20, b"\x00\x00")
                elif t == 3:                                  # PUBLISH
                    qos, retain = (h >> 1) & 3, h & 1
                    topic, i = _str(body, 0)
                    if qos:
                        pid = body[i:i + 2]; i += 2
                        self.send(0x40, pid)                  # PUBACK
                    self.b.publish(topic, body[i:], qos, retain)
                elif t == 8:                                  # SUBSCRIBE
                    pid, i, codes = body[:2], 2, bytearray()
                    new = []
                    while i < len(body):
                        f, i = _str(body, i)
                        q = min(body[i], 1); i += 1
                        self.subs[f.decode()] = q; codes.append(q); new.append(f.decode())
                    self.send(0x90, pid + bytes(codes))
                    for f in new:                             # retained messages
                        for topic, (payload, q) in self.b.retained.items():
                            if _match(f, topic.decode()):
                                self.deliver(topic, payload, min(q, self.subs[f]), True)
                elif t == 10:                                 # UNSUBSCRIBE
                    i = 2
                    while i < len(body):
                        f, i = _str(body, i); self.subs.pop(f.decode(), None)
                    self.send(0xB0, body[:2])
                elif t == 12:                                 # PINGREQ
                    self.send(0xD0, b"")
                elif t == 14:                                 # DISCONNECT
                    break
                # PUBACK (4) from clients is accepted and ignored
                await self.w.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.b.sessions.discard(self)
            self.w.close()


class Broker:
    def __init__(self):
        self.sessions = set()
        self.retained = {}

    def publish(self, topic: bytes, payload: bytes, qos: int, retain: bool):
        if retain:
            if payload:
                self.retained[topic] = (payload, qos)
            else:
                self.retained.pop(topic, None)
        ts = topic.decode()
        for s in list(self.sessions):
            q = max((sq for f, sq in s.subs.items() if _match(f, ts)), default=None)
            if q is not None:
                s.deliver(topic, payload, min(q, qos))

    async def _client(self, r, w):
        s = Session(self, r, w)
        self.sessions.add(s)
        await s.run()

    async def serve(self, host="127.0.0.1", port=1883):
        srv = await asyncio.start_server(self._client, host, port)
        async with srv:
            await srv.serve_forever()


def start_in_thread(host="127.0.0.1", port=1883):
    """Run a broker in a daemon thread (for tests and experiments)."""
    import threading
    import time

    def run():
        asyncio.run(Broker().serve(host, port))
    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1883)
    a = ap.parse_args()
    asyncio.run(Broker().serve(a.host, a.port))
