#!/usr/bin/env python3
import argparse
import math
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.raw_imu import Imu as RawImu
from mini_bdx_runtime.imu import Imu as FusionImu
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector

# =========================
# Configuration
# =========================
SERIAL_BY_ID = "/dev/ttyACM0"

CONTROL_HZ = 50.0
TURN_ON = True

# If True, initialize all optional peripherals in duck config so startup
# matches RL runtime hardware usage as closely as possible.
ENABLE_OPTIONAL_FEATURES = True

# Probe-only devices: data is read/logged but never used for control.
PROBE_RAW_IMU = True
PROBE_FUSION_IMU = False
PROBE_FEET_CONTACTS = False

# Keep probes lightweight; run every N control loops.
PROBE_EVERY_N_LOOPS = 1

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


def parse_args():
    parser = argparse.ArgumentParser(description="Policy-like stress with stage-level I/O diagnostics")
    parser.add_argument("--serial", type=str, default=SERIAL_BY_ID, help="Rustypot serial path")
    parser.add_argument("--duck_config_path", type=str, default=None, help="Optional duck config path")
    parser.add_argument("--control_hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--turn_on", action=argparse.BooleanOptionalAction, default=TURN_ON)
    parser.add_argument("--motion_mode", choices=["hold", "gentle"], default=MOTION_MODE)
    parser.add_argument("--motion_amplitude_rad", type=float, default=MOTION_AMPLITUDE_RAD)
    parser.add_argument("--motion_freq_hz", type=float, default=MOTION_FREQ_HZ)
    parser.add_argument("--extra_read_burst", type=int, choices=[0, 1, 2], default=EXTRA_READ_BURST)
    parser.add_argument("--fail_fast_consecutive", type=int, default=FAIL_FAST_CONSECUTIVE)
    parser.add_argument("--log_every_n_ok", type=int, default=LOG_EVERY_N_OK)
    parser.add_argument("--probe_every_n_loops", type=int, default=PROBE_EVERY_N_LOOPS)
    parser.add_argument("--enable_optional_features", action=argparse.BooleanOptionalAction, default=ENABLE_OPTIONAL_FEATURES)
    parser.add_argument("--probe_raw_imu", action=argparse.BooleanOptionalAction, default=PROBE_RAW_IMU)
    parser.add_argument("--probe_fusion_imu", action=argparse.BooleanOptionalAction, default=PROBE_FUSION_IMU)
    parser.add_argument("--probe_feet_contacts", action=argparse.BooleanOptionalAction, default=PROBE_FEET_CONTACTS)
    parser.add_argument("--auto_phases", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase_duration_s", type=float, default=90.0)
    return parser.parse_args()


def stage_ok(stats, stage, took):
    stats[stage]["ok"] += 1
    stats[stage]["last_ok_took_ms"] = round(took * 1000.0, 2)


def stage_fail(stats, stage, exc):
    stats[stage]["fail"] += 1
    stats[stage]["last_exc_type"] = type(exc).__name__
    stats[stage]["last_exc_msg"] = str(exc)


def run_stage(stats, stage, fn, *args, **kwargs):
    t0 = time.monotonic()
    try:
        out = fn(*args, **kwargs)
        stage_ok(stats, stage, time.monotonic() - t0)
        return out, None, None
    except Exception as exc:
        stage_fail(stats, stage, exc)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return None, exc, tb


def read_positions_strict(hwi):
    raw = hwi.io.read_present_position(list(hwi.joints.values()))
    return [
        pos - hwi.joints_offsets[joint]
        for joint, pos in zip(hwi.joints.keys(), raw)
    ]


def read_velocities_strict(hwi):
    raw = hwi.io.read_present_velocity(list(hwi.joints.values()))
    return [
        vel
        for joint, vel in zip(hwi.joints.keys(), raw)
    ]


def run_single_phase(args, phase_name="single"):
    serial_path = args.serial
    control_hz = args.control_hz
    motion_mode = args.motion_mode
    motion_amp = args.motion_amplitude_rad
    motion_freq = args.motion_freq_hz

    global SERIAL_BY_ID, MOTION_MODE, MOTION_AMPLITUDE_RAD, MOTION_FREQ_HZ
    SERIAL_BY_ID = serial_path
    MOTION_MODE = motion_mode
    MOTION_AMPLITUDE_RAD = motion_amp
    MOTION_FREQ_HZ = motion_freq

    log(f"PHASE_START name={phase_name}")
    log(
        f"CONFIG phase={phase_name} port={serial_path} hz={control_hz} turn_on={args.turn_on} "
        f"motion_mode={motion_mode} amp={motion_amp} freq={motion_freq} "
        f"extra_read_burst={args.extra_read_burst} probe_every_n_loops={args.probe_every_n_loops} "
        f"probe_raw_imu={args.probe_raw_imu} probe_fusion_imu={args.probe_fusion_imu} "
        f"probe_feet_contacts={args.probe_feet_contacts} enable_optional_features={args.enable_optional_features}"
    )

    duck_config = DuckConfig(config_json_path=args.duck_config_path, ignore_default=True)
    hwi = HWI(duck_config, usb_port=serial_path)

    raw_imu = None
    fusion_imu = None
    feet_contacts = None
    optional_features = {}

    stage_stats = defaultdict(lambda: defaultdict(int))

    for stage, ctor in (
        ("init.raw_imu", lambda: RawImu(sampling_freq=int(control_hz), upside_down=duck_config.imu_upside_down)),
        ("init.fusion_imu", lambda: FusionImu(sampling_freq=int(control_hz), upside_down=duck_config.imu_upside_down)),
        ("init.feet_contacts", FeetContacts),
        ("init.eyes", Eyes),
        ("init.projector", Projector),
        ("init.antennas", Antennas),
        ("init.sounds", lambda: Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets")),
    ):
        should_init = False
        if stage == "init.raw_imu":
            should_init = args.probe_raw_imu
        elif stage == "init.fusion_imu":
            should_init = args.probe_fusion_imu
        elif stage == "init.feet_contacts":
            should_init = args.probe_feet_contacts
        elif args.enable_optional_features:
            if stage == "init.eyes":
                should_init = duck_config.eyes
            elif stage == "init.projector":
                should_init = duck_config.projector
            elif stage == "init.antennas":
                should_init = duck_config.antennas
            elif stage == "init.sounds":
                should_init = duck_config.speaker

        if not should_init:
            continue

        obj, err, tb = run_stage(stage_stats, stage, ctor)
        if err is not None:
            log(f"EXCEPTION stage={stage} type={type(err).__name__} msg={err}")
            if tb:
                log(tb)
            continue

        if stage == "init.raw_imu":
            raw_imu = obj
        elif stage == "init.fusion_imu":
            fusion_imu = obj
        elif stage == "init.feet_contacts":
            feet_contacts = obj
        elif stage == "init.eyes":
            optional_features["eyes"] = obj
        elif stage == "init.projector":
            optional_features["projector"] = obj
        elif stage == "init.antennas":
            optional_features["antennas"] = obj
        elif stage == "init.sounds":
            optional_features["sounds"] = obj

    if args.turn_on:
        log("Calling hwi.turn_on()")
        hwi.turn_on()
        time.sleep(1.0)

    period = 1.0 / control_hz
    t0 = time.monotonic()

    cycle_idx = 0
    ok_count = 0
    fail_count = 0
    consecutive_fail = 0
    overrun_count = 0
    last_failure_stage = None

    last_exists, last_realpath = get_serial_state()
    log(f"SERIAL_INIT exists={last_exists} realpath={last_realpath}")

    while True:
        loop_start = time.monotonic()
        sim_t = loop_start - t0

        if args.phase_duration_s > 0 and sim_t >= args.phase_duration_s:
            log(f"PHASE_TIMEOUT name={phase_name} duration_s={args.phase_duration_s}")
            break

        exists, realpath = get_serial_state()
        if exists != last_exists or realpath != last_realpath:
            log(
                f"SERIAL_CHANGE old_exists={last_exists} old_realpath={last_realpath} "
                f"new_exists={exists} new_realpath={realpath}"
            )
            last_exists, last_realpath = exists, realpath

        try:
            pos, pos_err, _ = run_stage(stage_stats, "hwi.read_present_position", read_positions_strict, hwi)
            if pos_err is not None:
                last_failure_stage = "hwi.read_present_position"
                raise pos_err

            vel, vel_err, _ = run_stage(stage_stats, "hwi.read_present_velocity", read_velocities_strict, hwi)
            if vel_err is not None:
                last_failure_stage = "hwi.read_present_velocity"
                raise vel_err

            # Optional extra communication load
            if args.extra_read_burst >= 1:
                _, err, _ = run_stage(stage_stats, "hwi.read_present_position.extra", read_positions_strict, hwi)
                if err is not None:
                    last_failure_stage = "hwi.read_present_position.extra"
                    raise err
            if args.extra_read_burst >= 2:
                _, err, _ = run_stage(stage_stats, "hwi.read_present_velocity.extra", read_velocities_strict, hwi)
                if err is not None:
                    last_failure_stage = "hwi.read_present_velocity.extra"
                    raise err

            if args.probe_every_n_loops > 0 and (cycle_idx % args.probe_every_n_loops == 0):
                if raw_imu is not None:
                    _, err, _ = run_stage(stage_stats, "raw_imu.get_data", raw_imu.get_data)
                    if err is not None:
                        last_failure_stage = "raw_imu.get_data"
                        raise err

                if fusion_imu is not None:
                    _, err, _ = run_stage(stage_stats, "fusion_imu.get_data", fusion_imu.get_data)
                    if err is not None:
                        last_failure_stage = "fusion_imu.get_data"
                        raise err

                if feet_contacts is not None:
                    _, err, _ = run_stage(stage_stats, "feet_contacts.get", feet_contacts.get)
                    if err is not None:
                        last_failure_stage = "feet_contacts.get"
                        raise err

            if pos is None or vel is None:
                fail_count += 1
                consecutive_fail += 1
                log(
                    f"FAIL fail_count={fail_count} consecutive_fail={consecutive_fail} "
                    f"pos_is_none={pos is None} vel_is_none={vel is None} "
                    f"serial_exists={exists} serial_realpath={realpath}"
                )
                if consecutive_fail >= args.fail_fast_consecutive:
                    log("STOP too many consecutive failures")
                    break
            else:
                targets = make_targets(hwi, sim_t)
                _, err, _ = run_stage(stage_stats, "hwi.write_goal_position", hwi.set_position_all, targets)
                if err is not None:
                    last_failure_stage = "hwi.write_goal_position"
                    raise err

                ok_count += 1
                consecutive_fail = 0

                if ok_count % args.log_every_n_ok == 0:
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
                f"stage={last_failure_stage} type={type(e).__name__} msg={e} "
                f"serial_exists={exists} serial_realpath={realpath}"
            )
            log(traceback.format_exc())
            if consecutive_fail >= args.fail_fast_consecutive:
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

        cycle_idx += 1

    # Clean shutdown for optional peripherals
    for name in ("antennas", "eyes", "projector"):
        feature = optional_features.get(name)
        if feature is not None:
            _, err, _ = run_stage(stage_stats, f"stop.{name}", feature.stop)
            if err is not None:
                log(f"EXCEPTION stage=stop.{name} type={type(err).__name__} msg={err}")

    if feet_contacts is not None:
        _, err, _ = run_stage(stage_stats, "stop.feet_contacts", feet_contacts.stop)
        if err is not None:
            log(f"EXCEPTION stage=stop.feet_contacts type={type(err).__name__} msg={err}")

    log(f"SUMMARY_START phase={phase_name}")
    log(f"SUMMARY ok_count={ok_count}")
    log(f"SUMMARY fail_count={fail_count}")
    log(f"SUMMARY overrun_count={overrun_count}")
    for stage in sorted(stage_stats.keys()):
        ok = stage_stats[stage].get("ok", 0)
        fail = stage_stats[stage].get("fail", 0)
        last_exc_type = stage_stats[stage].get("last_exc_type", "")
        last_exc_msg = stage_stats[stage].get("last_exc_msg", "")
        last_ok_took_ms = stage_stats[stage].get("last_ok_took_ms", "")
        log(
            f"SUMMARY_STAGE stage={stage} ok={ok} fail={fail} "
            f"last_ok_took_ms={last_ok_took_ms} "
            f"last_exc_type={last_exc_type} last_exc_msg={last_exc_msg}"
        )
    exists, realpath = get_serial_state()
    log(f"SUMMARY final_serial_exists={exists} final_serial_realpath={realpath}")
    log(f"PHASE_END name={phase_name}")

    stage_failures = {stage: vals.get("fail", 0) for stage, vals in stage_stats.items() if vals.get("fail", 0) > 0}
    first_failure_stage = "none"
    if len(stage_failures) > 0:
        first_failure_stage = sorted(stage_failures.keys())[0]

    return {
        "phase": phase_name,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "overrun_count": overrun_count,
        "first_failure_stage": first_failure_stage,
        "stage_failures": stage_failures,
    }


def make_phase_args(args, **overrides):
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def print_phase_comparison(results):
    log("PHASE_COMPARISON_START")
    for result in results:
        log(
            f"PHASE_RESULT name={result['phase']} ok={result['ok_count']} "
            f"fail={result['fail_count']} overrun={result['overrun_count']} "
            f"first_failure_stage={result['first_failure_stage']}"
        )
        if result["stage_failures"]:
            for stage, count in sorted(result["stage_failures"].items()):
                log(f"PHASE_RESULT_STAGE name={result['phase']} stage={stage} fail_count={count}")
    log("PHASE_COMPARISON_END")


def main():
    args = parse_args()

    if not args.auto_phases:
        run_single_phase(args, phase_name="single")
        return

    phase_results = []
    phases = [
        (
            "serial_only_hold",
            {
                "motion_mode": "hold",
                "extra_read_burst": 0,
                "probe_raw_imu": False,
                "probe_fusion_imu": False,
                "probe_feet_contacts": False,
                "enable_optional_features": False,
            },
        ),
        (
            "serial_stress_hold",
            {
                "motion_mode": "hold",
                "extra_read_burst": 2,
                "probe_raw_imu": False,
                "probe_fusion_imu": False,
                "probe_feet_contacts": False,
                "enable_optional_features": False,
            },
        ),
        (
            "serial_plus_raw_imu",
            {
                "motion_mode": "hold",
                "extra_read_burst": 0,
                "probe_raw_imu": True,
                "probe_fusion_imu": False,
                "probe_feet_contacts": False,
            },
        ),
        (
            "policy_like_gentle",
            {
                "motion_mode": "gentle",
                "motion_amplitude_rad": min(args.motion_amplitude_rad, 0.03),
                "extra_read_burst": args.extra_read_burst,
                "probe_raw_imu": args.probe_raw_imu,
                "probe_fusion_imu": args.probe_fusion_imu,
                "probe_feet_contacts": args.probe_feet_contacts,
                "enable_optional_features": args.enable_optional_features,
            },
        ),
    ]

    for phase_name, overrides in phases:
        phase_args = make_phase_args(args, **overrides)
        result = run_single_phase(phase_args, phase_name=phase_name)
        phase_results.append(result)

    print_phase_comparison(phase_results)


if __name__ == "__main__":
    main()
