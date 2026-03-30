#!/usr/bin/env python3
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
logfile = LOGDIR / f"pi_power_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"

with logfile.open("w") as f:
    f.write(f"=== START {datetime.now().isoformat()} ===\n")
    f.flush()

    while True:
        ts = datetime.now().isoformat()
        throttled = run(["vcgencmd", "get_throttled"])
        temp = run(["vcgencmd", "measure_temp"])
        line = f"{ts} | {throttled} | {temp}"
        print(line)
        f.write(line + "\n")
        f.flush()
        time.sleep(0.5)