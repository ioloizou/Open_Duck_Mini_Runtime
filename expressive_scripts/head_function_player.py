#!/usr/bin/env python3
"""
Expressive head controller using attractor-tracked emotional postures.

Key change:
- Oscillations are added to the GOAL, not as acceleration forcing.

So for each joint:
    g_eff(t) = g_base + oscillator(t)
    q_ddot = k * (g_eff - q) - c * q_dot

This makes oscillation amplitudes much easier to interpret and tune.
"""

import argparse
import os
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")

# =========================
# CONFIG
# =========================
DUCK_CONFIG_PATH = f"{HOME_DIR}/duck_config.json"
SERIAL_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082643-if00"

FREQ_HZ = 50
DURATION_S = 0.0  # 0.0 = run forever
PRINT_EVERY_S = 10
CLIP_TO_LIMITS = True
ENABLE_ANTENNAS = True

BODY_KP = 30.0
HEAD_KP = 8.0
KD = 0.0

CURRENT_EMOTION = "happy"
# Options: happy, sad, curious, intrigued

JOINT_LIMITS = {
    "neck_pitch": (-0.34, 1.10),
    "head_pitch": (-0.78, 0.30),
    "head_yaw": (-0.50, 0.50),
    "head_roll": (-0.50, 0.50),
}

ANTENNA_LIMITS = (-1.0, 1.0)


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Oscillator:
    amplitude: float = 0.0
    frequency_hz: float = 0.0
    phase: float = 0.0

    def value(self, t: float) -> float:
        if self.amplitude == 0.0 or self.frequency_hz == 0.0:
            return 0.0
        return self.amplitude * math.sin(
            2.0 * math.pi * self.frequency_hz * t + self.phase
        )


@dataclass
class JointControl:
    goal_offset: float = 0.0
    osc: Oscillator = field(default_factory=Oscillator)


@dataclass
class AntennaControl:
    offset_left: float = 0.0
    offset_right: float = 0.0
    osc_left: Oscillator = field(default_factory=Oscillator)
    osc_right: Oscillator = field(default_factory=Oscillator)


@dataclass
class EmotionSpec:
    k: float
    c: float
    joints: Dict[str, JointControl]
    antennas: AntennaControl
    description: str = ""


@dataclass
class JointState:
    q: float
    v: float = 0.0


# =========================
# EMOTIONS
# =========================
def make_intrigued_roll() -> float:
    return random.choice([-1.0, 1.0]) * 0.3


EMOTIONS: Dict[str, EmotionSpec] = {
    "happy": EmotionSpec(
        k=200.0,
        c=7.0,
        description="Looks slightly up, cheerful head-roll wiggle, active antennas.",
        joints={
            "neck_pitch": JointControl(
                goal_offset=0.08,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_pitch": JointControl(
                goal_offset=0.10,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_yaw": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.15, 2, 0.0),
            ),
            "head_roll": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.25, 2.5, 0.0),  # much more visible now
            ),
        },
        antennas=AntennaControl(
            offset_left=0.15,
            offset_right=0.15,
            osc_left=Oscillator(0.1, 5, 0.0),
            osc_right=Oscillator(0.1, 5, 0.0),
        ),
    ),
    "sad": EmotionSpec(
        k=20.0,
        c=4.5,
        description="Slow downward collapse with little or no rhythmic motion.",
        joints={
            "neck_pitch": JointControl(
                goal_offset=-0.3,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_pitch": JointControl(
                goal_offset=-0.7,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_yaw": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.1, 1, 0.0),
            ),
            "head_roll": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
        },
        antennas=AntennaControl(
            offset_left=-0.35,
            offset_right=-0.35,
            osc_left=Oscillator(0.00, 0.00, 0.0),
            osc_right=Oscillator(0.00, 0.00, 0.0),
        ),
    ),
    "curious": EmotionSpec(
        k=30.0,
        c=6.0,
        description="Slightly looks up and scans slowly side to side.",
        joints={
            "neck_pitch": JointControl(
                goal_offset=0.05,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_pitch": JointControl(
                goal_offset=0.07,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_yaw": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.35, 0.75, 0.0),
            ),
            "head_roll": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
        },
        antennas=AntennaControl(
            offset_left=0.30,
            offset_right=0.30,
            osc_left=Oscillator(0.04, 0.30, 0.0),
            osc_right=Oscillator(0.04, 0.30, 0.0),
        ),
    ),
    "intrigued": EmotionSpec(
        k=6.0,
        c=5.0,
        description="Small upward posture with a slow tilted head to one side.",
        joints={
            "neck_pitch": JointControl(
                goal_offset=0.04,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_pitch": JointControl(
                goal_offset=0.08,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_yaw": JointControl(
                goal_offset=0.00,
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
            "head_roll": JointControl(
                goal_offset=make_intrigued_roll(),
                osc=Oscillator(0.00, 0.00, 0.0),
            ),
        },
        antennas=AntennaControl(
            offset_left=1,
            offset_right=1,
            osc_left=Oscillator(0.00, 0.00, 0.0),
            osc_right=Oscillator(0.00, 0.00, 0.0),
        ),
    ),
}


# =========================
# HELPERS
# =========================
def clip_if_needed(joint_name: str, value: float, use_clip: bool) -> float:
    if not use_clip:
        return value
    low, high = JOINT_LIMITS[joint_name]
    return float(np.clip(value, low, high))


def clip_antenna(value: float, use_clip: bool) -> float:
    if not use_clip:
        return value
    low, high = ANTENNA_LIMITS
    return float(np.clip(value, low, high))


def attractor_step(
    q: float, v: float, g: float, dt: float, k: float, c: float
) -> tuple[float, float]:
    """
    Semi-implicit Euler integration of:
        q_ddot = k * (g - q) - c * q_dot
    """
    a = k * (g - q) - c * v
    v_new = v + dt * a
    q_new = q + dt * v_new
    return q_new, v_new


def build_joint_goals(
    init_pose: Dict[str, float], emotion: EmotionSpec
) -> Dict[str, float]:
    goals: Dict[str, float] = {}
    for joint_name, joint_ctrl in emotion.joints.items():
        goals[joint_name] = init_pose[joint_name] + joint_ctrl.goal_offset
    return goals


def compute_antennas(
    emotion: EmotionSpec, t: float, use_clip: bool
) -> tuple[float, float]:
    ant = emotion.antennas
    left = ant.offset_left + ant.osc_left.value(t)
    right = ant.offset_right + ant.osc_right.value(t)
    return clip_antenna(left, use_clip), clip_antenna(right, use_clip)


def get_emotion_spec(name: str) -> EmotionSpec:
    if name not in EMOTIONS:
        valid = ", ".join(sorted(EMOTIONS.keys()))
        raise ValueError(f"Unknown emotion '{name}'. Valid options: {valid}")
    return EMOTIONS[name]


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play an expressive head motion for a selected emotion."
    )
    parser.add_argument(
        "emotion",
        nargs="?",
        default=CURRENT_EMOTION,
        choices=sorted(EMOTIONS.keys()),
        help="Emotion to play.",
    )
    args = parser.parse_args()
    selected_emotion = args.emotion

    duck_config = DuckConfig(config_json_path=DUCK_CONFIG_PATH)
    hwi = HWI(duck_config, usb_port=SERIAL_PORT)
    antennas = Antennas() if (ENABLE_ANTENNAS and duck_config.antennas) else None

    kps = [BODY_KP] * 14
    kds = [KD] * 14
    kps[5:9] = [HEAD_KP, HEAD_KP, HEAD_KP, HEAD_KP]

    hwi.set_kps(kps)
    hwi.set_kds(kds)
    hwi.turn_on()

    init = hwi.init_pos
    dt = 1.0 / FREQ_HZ
    use_clip = CLIP_TO_LIMITS

    emotion = get_emotion_spec(selected_emotion)
    goals = build_joint_goals(init, emotion)

    joint_states: Dict[str, JointState] = {
        joint_name: JointState(q=float(init[joint_name]), v=0.0)
        for joint_name in emotion.joints.keys()
    }

    start_t = time.time()
    next_print_t = start_t

    print("Starting expressive head controller")
    print(f"Emotion: {selected_emotion}")
    print(f"Description: {emotion.description}")
    print(f"Clipping: {'on' if use_clip else 'off'}")
    print(f"k={emotion.k:.2f}, c={emotion.c:.2f}")

    try:
        while True:
            loop_t = time.time()
            t = loop_t - start_t

            if DURATION_S > 0.0 and t >= DURATION_S:
                break

            targets: Dict[str, float] = {}

            for joint_name, joint_ctrl in emotion.joints.items():
                state = joint_states[joint_name]

                g_base = goals[joint_name]
                g_eff = g_base + joint_ctrl.osc.value(t)

                q_new, v_new = attractor_step(
                    q=state.q,
                    v=state.v,
                    g=g_eff,
                    dt=dt,
                    k=emotion.k,
                    c=emotion.c,
                )

                q_new = clip_if_needed(joint_name, q_new, use_clip)

                low, high = JOINT_LIMITS[joint_name]
                if use_clip and (abs(q_new - low) < 1e-9 or abs(q_new - high) < 1e-9):
                    v_new = 0.0

                state.q = q_new
                state.v = v_new
                targets[joint_name] = q_new

            antenna_left, antenna_right = compute_antennas(emotion, t, use_clip)

            for joint_name, value in targets.items():
                hwi.set_position(joint_name, value)

            if antennas is not None:
                antennas.set_position_left(antenna_left)
                antennas.set_position_right(antenna_right)

            if loop_t >= next_print_t:
                print(
                    "t={:.2f}s emotion={} neck={:+.3f} hp={:+.3f} hy={:+.3f} hr={:+.3f} "
                    "Lant={:+.3f} Rant={:+.3f}".format(
                        t,
                        selected_emotion,
                        targets["neck_pitch"],
                        targets["head_pitch"],
                        targets["head_yaw"],
                        targets["head_roll"],
                        antenna_left,
                        antenna_right,
                    )
                )
                next_print_t = loop_t + max(0.05, PRINT_EVERY_S)

            took = time.time() - loop_t
            time.sleep(max(0.0, dt - took))

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if antennas is not None:
            antennas.stop()
        print("Turning off torque")
        hwi.turn_off()


if __name__ == "__main__":
    main()
