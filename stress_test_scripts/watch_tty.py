#!/usr/bin/env python3
import glob
import os
import time
from datetime import datetime
from pathlib import Path

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
logfile = LOGDIR / f"tty_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def list_ports():
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    info = []
    for p in ports:
        try:
            real = os.path.realpath(p)
        except Exception:
            real = "?"
        info.append((p, real))
    return info

last = None
with logfile.open("w") as f:
    while True:
        now = list_ports()
        if now != last:
            ts = datetime.now().isoformat()
            line = f"{ts} | {now}"
            print(line)
            f.write(line + "\n")
            f.flush()
            last = now
        time.sleep(0.2)