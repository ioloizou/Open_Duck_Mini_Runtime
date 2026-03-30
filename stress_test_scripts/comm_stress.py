#!/usr/bin/env python3
import time
import traceback
from datetime import datetime
from pathlib import Path

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

READ_HZ = 100   # try 20 → 50 → 100 → 200

duck_config = DuckConfig()
hwi = HWI(duck_config)

logfile = LOGDIR / f"comm_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.now().isoformat()
    line = f"{ts} | {msg}"
    print(line)
    with logfile.open("a") as f:
        f.write(line + "\n")

def main():
    log(f"START comm stress READ_HZ={READ_HZ}")

    hwi.turn_on()
    time.sleep(1)

    period = 1.0 / READ_HZ
    count = 0
    t0 = time.time()

    while True:
        t_start = time.time()

        try:
            pos = hwi.get_present_positions()
            vel = hwi.get_present_velocities()

            if pos is None:
                log("WARNING: pos None")
            if vel is None:
                log("WARNING: vel None")

            count += 1
            if count % 50 == 0:
                rate = count / (time.time() - t0)
                log(f"ok count={count} rate={rate:.1f}Hz")

        except Exception as e:
            log(f"ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            break

        dt = time.time() - t_start
        sleep = period - dt
        if sleep > 0:
            time.sleep(sleep)

if __name__ == "__main__":
    main()