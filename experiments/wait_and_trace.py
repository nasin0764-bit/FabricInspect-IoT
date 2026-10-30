"""Wait (up to 90 min) until the host is unthrottled, then measure the traces once."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
from run_iot import speed_check

deadline = time.time() + 90 * 60
while time.time() < deadline:
    if speed_check()["ok"]:
        sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "run_traces.py")]))
    time.sleep(300)
print("host stayed throttled for 90 min", flush=True)
sys.exit(2)
