#!/usr/bin/env python3
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig


# =========================
# Configuration
# =========================
SERIAL_BY_ID = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082643-if00"

CONTROL_HZ = 50.0
TURN_ON = True

# Motion modes:
#   "hold"   -> no motion, just reads + rewrite init pose
#   "gentle" -> very small slow oscillation on a few joints
MOTION_MODE = "gentle"

# Very small safe motion defaults
MOTION_AMPLITUDE_RAD = 0.05   # ~2.9 deg
MOTION_FREQ_HZ = 0.25         # very slow

# Only use safer leg pitch/knee/ankle pairings, symmetrically
MOTION_JOINTS = [
    "left_hip_pitch",
    "right_hip_pitch",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Number of extra comm operations per cycle to increase stress
# 0 = controller-like baseline
# 1 = one extra position read
# 2 = one extra position + one extra velocity read
EXTRA_READ_BURST = 0

FAIL_FAST_CONSECUTIVE = 5
LOG_EVERY_N_OK = 1000

LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
LOGFILE = LOGDIR / f"policy_like_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg):
    ts = datetime.now().isoformat()
    line = f"{ts} | {msg}"
    print(line, flush=True)
    with LOGFILE.open("a") as f:
        f.write(line + "\n")


def get_serial_state():
    exists = os.path.exists(SERIAL_BY_ID)
    realpath = os.path.realpath(SERIAL_BY_ID) if exists else None
    return exists, realpath


def make_targets(hwi: HWI, t: float):
    """
    Returns a full joint dict to pass to set_position_all().
    Keeps all joints at init_pos unless selected for tiny oscillation.
    """
    targets = dict(hwi.init_pos)

    if MOTION_MODE == "hold":
        return targets

    phase = math.sin(2.0 * math.pi * MOTION_FREQ_HZ * t)

    # Small symmetric motion to avoid abrupt posture changes.
    # Signs chosen to preserve rough bilateral symmetry.
    offsets = {
        "left_hip_pitch": -MOTION_AMPLITUDE_RAD * phase,
        "right_hip_pitch": +MOTION_AMPLITUDE_RAD * phase,
        "left_knee": +MOTION_AMPLITUDE_RAD * phase,
        "right_knee": +MOTION_AMPLITUDE_RAD * phase,
        "left_ankle": -0.5 * MOTION_AMPLITUDE_RAD * phase,
        "right_ankle": -0.5 * MOTION_AMPLITUDE_RAD * phase,
    }

    for joint in MOTION_JOINTS:
        if joint in offsets:
            targets[joint] = hwi.init_pos[joint] + offsets[joint]

    return targets


def summarize(arr, n=4):
    if arr is None:
        return "None"
    vals = list(arr)
    if len(vals) <= n:
        return str(vals)
    return str(vals[:n])[:-1] + ", ...]"


def main():
    log("SCRIPT_START")
    log(
        f"CONFIG port={SERIAL_BY_ID} hz={CONTROL_HZ} turn_on={TURN_ON} "
        f"motion_mode={MOTION_MODE} amp={MOTION_AMPLITUDE_RAD} freq={MOTION_FREQ_HZ} "
        f"extra_read_burst={EXTRA_READ_BURST}"
    )

    duck_config = DuckConfig()
    hwi = HWI(duck_config, usb_port=SERIAL_BY_ID)

    if TURN_ON:
        log("Calling hwi.turn_on()")
        hwi.turn_on()
        time.sleep(1.0)

    period = 1.0 / CONTROL_HZ
    t0 = time.monotonic()

    ok_count = 0
    fail_count = 0
    consecutive_fail = 0
    overrun_count = 0

    last_exists, last_realpath = get_serial_state()
    log(f"SERIAL_INIT exists={last_exists} realpath={last_realpath}")

    while True:
        loop_start = time.monotonic()
        sim_t = loop_start - t0

        exists, realpath = get_serial_state()
        if exists != last_exists or realpath != last_realpath:
            log(
                f"SERIAL_CHANGE old_exists={last_exists} old_realpath={last_realpath} "
                f"new_exists={exists} new_realpath={realpath}"
            )
            last_exists, last_realpath = exists, realpath

        try:
            # Controller-like baseline
            pos = hwi.get_present_positions()
            vel = hwi.get_present_velocities()

            # Optional extra communication load
            if EXTRA_READ_BURST >= 1:
                _ = hwi.get_present_positions()
            if EXTRA_READ_BURST >= 2:
                _ = hwi.get_present_velocities()

            if pos is None or vel is None:
                fail_count += 1
                consecutive_fail += 1
                log(
                    f"FAIL fail_count={fail_count} consecutive_fail={consecutive_fail} "
                    f"pos_is_none={pos is None} vel_is_none={vel is None} "
                    f"serial_exists={exists} serial_realpath={realpath}"
                )
                if consecutive_fail >= FAIL_FAST_CONSECUTIVE:
                    log("STOP too many consecutive failures")
                    break
            else:
                targets = make_targets(hwi, sim_t)
                hwi.set_position_all(targets)

                ok_count += 1
                consecutive_fail = 0

                if ok_count % LOG_EVERY_N_OK == 0:
                    log(
                        f"OK ok_count={ok_count} fail_count={fail_count} "
                        f"pos_sample={summarize(pos)} vel_sample={summarize(vel)} "
                        f"serial_exists={exists} serial_realpath={realpath}"
                    )

        except KeyboardInterrupt:
            log("KeyboardInterrupt")
            break
        except Exception as e:
            fail_count += 1
            consecutive_fail += 1
            log(
                f"EXCEPTION fail_count={fail_count} consecutive_fail={consecutive_fail} "
                f"type={type(e).__name__} msg={e}"
            )
            log(traceback.format_exc())
            if consecutive_fail >= FAIL_FAST_CONSECUTIVE:
                log("STOP too many consecutive exceptions")
                break

        took = time.monotonic() - loop_start
        slack = period - took
        if slack < 0:
            overrun_count += 1
            log(
                f"OVERRUN overrun_count={overrun_count} "
                f"loop_time={took:.4f}s budget={period:.4f}s excess={-slack:.4f}s"
            )
        else:
            time.sleep(slack)

    log("SUMMARY_START")
    log(f"SUMMARY ok_count={ok_count}")
    log(f"SUMMARY fail_count={fail_count}")
    log(f"SUMMARY overrun_count={overrun_count}")
    exists, realpath = get_serial_state()
    log(f"SUMMARY final_serial_exists={exists} final_serial_realpath={realpath}")
    log("SCRIPT_END")


if __name__ == "__main__":
    main()
