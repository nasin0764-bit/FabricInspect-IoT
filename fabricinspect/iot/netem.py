"""Application-level link emulation for experiments on a single host.

A Link serialises outgoing messages through a FIFO with a fixed bit-rate and
adds a one-way propagation delay, i.e. a message of n bytes sent at time t is
handed to the MQTT client at  max(t, link_free) + 8n/rate + delay.
Incoming messages are delayed by the same one-way delay (downlink bandwidth
is assumed ample: results are < 200 B). rate=None disables emulation.
"""
import heapq
import threading
import time

PROFILES = {
    "lan": dict(rate=None, delay=0.0),                  # wired / loopback
    "wifi": dict(rate=20e6, delay=0.004),              # loaded industrial Wi-Fi
    "constrained": dict(rate=1e6, delay=0.040),        # weak Wi-Fi / cellular IoT uplink
}


class Link:
    def __init__(self, rate=None, delay=0.0):
        self.rate, self.delay = rate, delay
        self._free = 0.0
        self._q = []
        self._cv = threading.Condition()
        self._n = 0
        self.bytes_up = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def send(self, nbytes: int, fn):
        """Schedule fn() after the emulated uplink transfer of nbytes."""
        now = time.perf_counter()
        self.bytes_up += nbytes
        if self.rate is None and not self.delay:
            return fn()
        with self._cv:
            start = max(now, self._free)
            tx = 8 * nbytes / self.rate if self.rate else 0.0
            self._free = start + tx
            self._n += 1
            heapq.heappush(self._q, (start + tx + self.delay, self._n, fn))
            self._cv.notify()

    def receive(self, fn):
        """Deliver an incoming message after the one-way delay."""
        if not self.delay:
            return fn()
        with self._cv:
            self._n += 1
            heapq.heappush(self._q, (time.perf_counter() + self.delay, self._n, fn))
            self._cv.notify()

    def _pump(self):
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                due, _, fn = self._q[0]
                wait = due - time.perf_counter()
                if wait > 0:
                    self._cv.wait(wait)
                    continue
                heapq.heappop(self._q)
            fn()
