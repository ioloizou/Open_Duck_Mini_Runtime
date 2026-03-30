#!/usr/bin/env python3
import time
import math
import traceback
from datetime import datetime
from pathlib import Path

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

duck_config = DuckConfig()
hwi = HWI(duck_config)

logfile = LOGDIR / f"motion_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.now().isoformat()
    line = f"{ts} | {msg}"
    print(line)
    with logfile.open("a") as f:
        f.write(line + "\n")

def main():
    log("START motion stress")

    hwi.turn_on()
    time.sleep(1)

    # start small → increase later
    amplitude = 0.2   # radians
    freq = 0.5        # Hz

    joints = ["left_knee", "right_knee"]  # start simple

    t0 = time.time()
    step = 0

    while True:
        t = time.time() - t0
        phase = math.sin(2 * math.pi * freq * t)

        try:
            for joint in joints:
                hwi.set_position(joint, amplitude * phase)

            step += 1
            if step % 50 == 0:
                log(f"ok step={step}")

        except Exception as e:
            log(f"ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            break

        time.sleep(0.02)

if __name__ == "__main__":
    main()