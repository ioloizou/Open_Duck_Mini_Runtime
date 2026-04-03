#!/usr/bin/env python3
import time
from datetime import datetime
from pathlib import Path

import rustypot

# ==== CONFIG ====
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082643-if00"

# Common Feetech baudrates
BAUDRATES = [
    1000000,
    500000,
    250000,
    115200,
    57600,
    38400,
]

# Try one known servo ID first
TEST_ID = 20  # change if needed

TRIES_PER_BAUD = 10
TIMEOUT_BETWEEN_TRIES = 0.05

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

logfile = LOGDIR / f"baud_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg):
    ts = datetime.now().isoformat()
    line = f"{ts} | {msg}"
    print(line)
    with logfile.open("a") as f:
        f.write(line + "\n")


def test_baud(baud):
    log(f"\n=== Testing baudrate {baud} ===")

    try:
        io = rustypot.feetech(PORT, baud)
        time.sleep(0.2)
    except Exception as e:
        log(f"FAILED to open port: {e}")
        return False

    success = 0
    fail = 0

    for i in range(TRIES_PER_BAUD):
        try:
            pos = io.read_present_position([TEST_ID])
            if pos is not None:
                success += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1

        time.sleep(TIMEOUT_BETWEEN_TRIES)

    log(f"RESULT baud={baud} success={success}/{TRIES_PER_BAUD}")

    return success > 0


def main():
    log("=== BAUDRATE SCAN START ===")

    working = []

    for baud in BAUDRATES:
        ok = test_baud(baud)
        if ok:
            working.append(baud)

    log("\n=== SUMMARY ===")
    if working:
        log(f"Working baudrates: {working}")
    else:
        log("No baudrate worked")

    log("=== END ===")


if __name__ == "__main__":
    main()
