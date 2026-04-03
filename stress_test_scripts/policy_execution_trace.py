#!/usr/bin/env python3
import argparse
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.rl_utils import LowPassActionFilter, make_action_dict
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.xbox_controller import XBoxController

HOME_DIR = os.path.expanduser("~")


class TraceLogger:
    def __init__(self, logdir: str):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = self.logdir / f"policy_execution_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(self, msg: str):
        ts = datetime.now().isoformat()
        line = f"{ts} | {msg}"
        print(line, flush=True)
        with self.logfile.open("a") as f:
            f.write(line + "\n")


class RLWalkTrace:
    def __init__(
        self,
        onnx_model_path: str,
        logger: TraceLogger,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=None,
        action_scale: float = 0.25,
        commands: bool = False,
        pitch_bias: float = 0,
        save_obs: bool = False,
        replay_obs=None,
        cutoff_frequency=None,
        strict_hwi_reads: bool = True,
        continue_on_error: bool = False,
    ) -> None:
        if pid is None:
            pid = [30, 0, 0]

        self.logger = logger
        self.serial_port = serial_port
        self.control_freq = control_freq
        self.pid = pid
        self.action_scale = action_scale
        self.commands = commands
        self.pitch_bias = pitch_bias
        self.save_obs = save_obs
        self.replay_obs = replay_obs
        self.strict_hwi_reads = strict_hwi_reads
        self.continue_on_error = continue_on_error

        self.duck_config = self.run_stage(
            "init.duck_config", DuckConfig, config_json_path=duck_config_path
        )

        self.onnx_model_path = onnx_model_path
        self.policy = self.run_stage(
            "init.onnx", OnnxInfer, self.onnx_model_path, awd=True
        )

        self.num_dofs = 14
        self.max_motor_velocity = 5.24

        if self.save_obs:
            self.saved_obs = []

        if self.replay_obs is not None:
            self.replay_obs = self.run_stage(
                "init.replay_obs", pickle.load, open(self.replay_obs, "rb")
            )

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(self.control_freq, cutoff_frequency)

        self.hwi = self.run_stage("init.hwi", HWI, self.duck_config, serial_port)

        self.run_stage("init.start", self.start)

        self.imu = self.run_stage(
            "init.imu",
            Imu,
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = self.try_stage("init.feet_contacts", FeetContacts)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20
        self.xbox_controller = None
        if self.commands:
            self.xbox_controller = self.run_stage(
                "init.xbox_controller", XBoxController, sampling_freq=self.command_freq
            )

        self.eyes = None
        self.projector = None
        self.sounds = None
        self.antennas = None

        if self.duck_config.eyes:
            self.eyes = self.try_stage("init.eyes", Eyes)
        if self.duck_config.projector:
            self.projector = self.try_stage("init.projector", Projector)
        if self.duck_config.speaker:
            self.sounds = self.try_stage(
                "init.sounds",
                Sounds,
                volume=1.0,
                sound_directory="../mini_bdx_runtime/assets",
            )
        if self.duck_config.antennas:
            self.antennas = self.try_stage("init.antennas", Antennas)

        self.logger.log(
            f"INIT_DONE strict_hwi_reads={self.strict_hwi_reads} continue_on_error={self.continue_on_error}"
        )

    def get_serial_state(self):
        exists = os.path.exists(self.serial_port)
        realpath = os.path.realpath(self.serial_port) if exists else None
        return exists, realpath

    def log_exception(self, stage: str, exc: Exception):
        exists, realpath = self.get_serial_state()
        self.logger.log(
            f"EXCEPTION stage={stage} type={type(exc).__name__} msg={exc} "
            f"serial_exists={exists} serial_realpath={realpath}"
        )
        self.logger.log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def run_stage(self, stage: str, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            out = fn(*args, **kwargs)
            dt = (time.monotonic() - t0) * 1000.0
            self.logger.log(f"STAGE_OK stage={stage} took_ms={dt:.2f}")
            return out
        except Exception as exc:
            self.log_exception(stage, exc)
            raise

    def try_stage(self, stage: str, fn, *args, **kwargs):
        try:
            return self.run_stage(stage, fn, *args, **kwargs)
        except Exception:
            return None

    def read_positions_strict(self):
        raw = self.hwi.io.read_present_position(list(self.hwi.joints.values()))
        present_positions = [
            pos - self.hwi.joints_offsets[joint]
            for joint, pos in zip(self.hwi.joints.keys(), raw)
        ]
        return np.array(np.around(present_positions, 3))

    def read_velocities_strict(self):
        raw = self.hwi.io.read_present_velocity(list(self.hwi.joints.values()))
        present_velocities = [
            vel
            for joint, vel in zip(self.hwi.joints.keys(), raw)
        ]
        return np.array(np.around(present_velocities, 3))

    def quat_apply_inverse(self, quat, vec):
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec)
        return vec - w * t + np.cross(xyz, t)

    def compute_projected_gravity(self, imu_data):
        gravity = np.array([0, 0, -1])
        return self.quat_apply_inverse(imu_data["orientation"], gravity)

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14
        kps[5:9] = [8] * 4

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
        time.sleep(2)

    def get_obs(self):
        imu_data = self.run_stage("obs.imu.get_imu_data", self.imu.get_imu_data)

        if self.strict_hwi_reads:
            dof_pos = self.run_stage("obs.hwi.read_positions", self.read_positions_strict)
            dof_vel = self.run_stage("obs.hwi.read_velocities", self.read_velocities_strict)
        else:
            dof_pos = self.run_stage("obs.hwi.get_present_positions", self.hwi.get_present_positions)
            dof_vel = self.run_stage("obs.hwi.get_present_velocities", self.hwi.get_present_velocities)

        if dof_pos is None or dof_vel is None:
            self.logger.log("OBS_SKIP reason=dof_none")
            return None

        if len(dof_pos) != self.num_dofs:
            self.logger.log(f"OBS_SKIP reason=bad_pos_len got={len(dof_pos)} expected={self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            self.logger.log(f"OBS_SKIP reason=bad_vel_len got={len(dof_vel)} expected={self.num_dofs}")
            return None

        cmds = self.last_commands

        projected_gravity = None
        if "quaternion" in imu_data and imu_data["quaternion"] is not None:
            imu_quat = imu_data["quaternion"]
            if len(imu_quat) == 4 and all(q is not None for q in imu_quat):
                try:
                    projected_gravity = self.run_stage(
                        "obs.compute_projected_gravity", self.compute_projected_gravity, imu_data
                    )
                except Exception:
                    projected_gravity = None

        if projected_gravity is None:
            projected_gravity = np.array([0.0, 0.0, -1.0])

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                projected_gravity,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                cmds,
            ]
        )
        return obs

    def shutdown(self):
        if self.antennas is not None:
            self.try_stage("shutdown.antennas", self.antennas.stop)
        if self.eyes is not None:
            self.try_stage("shutdown.eyes", self.eyes.stop)
        if self.projector is not None:
            self.try_stage("shutdown.projector", self.projector.stop)

    def run(self, max_steps: int = 0):
        i = 0
        failure_count = 0
        start_t = time.time()
        self.logger.log("RUN_START")

        try:
            while True:
                if max_steps > 0 and i >= max_steps:
                    self.logger.log(f"RUN_STOP reason=max_steps max_steps={max_steps}")
                    break

                left_trigger = 0
                right_trigger = 0
                t = time.time()

                try:
                    if self.commands:
                        out = self.run_stage(
                            "loop.commands.read", self.xbox_controller.get_last_command
                        )
                        self.last_commands, self.buttons, left_trigger, right_trigger = out

                        if self.buttons.X.triggered and self.projector is not None:
                            self.run_stage("loop.commands.projector.switch", self.projector.switch)

                        if self.buttons.B.triggered and self.sounds is not None:
                            self.run_stage("loop.commands.sounds.play_random", self.sounds.play_random_sound)

                        if self.antennas is not None:
                            self.run_stage("loop.commands.antennas.left", self.antennas.set_position_left, right_trigger)
                            self.run_stage("loop.commands.antennas.right", self.antennas.set_position_right, left_trigger)

                        if self.buttons.A.triggered:
                            self.paused = not self.paused
                            self.logger.log(f"PAUSE_STATE paused={self.paused}")

                    if self.paused:
                        time.sleep(0.1)
                        continue

                    obs = self.get_obs()
                    if obs is None:
                        continue

                    if self.save_obs:
                        self.saved_obs.append(obs)

                    if self.replay_obs is not None:
                        if i >= len(self.replay_obs):
                            self.logger.log("RUN_STOP reason=replay_end")
                            break
                        obs = self.replay_obs[i]

                    action = self.run_stage("loop.policy.infer", self.policy.infer, obs)
                    self.last_action = action.copy()

                    self.motor_targets = self.init_pos + action * self.action_scale

                    if self.action_filter is not None:
                        self.run_stage("loop.filter.push", self.action_filter.push, self.motor_targets)
                        filtered_motor_targets = self.run_stage(
                            "loop.filter.get", self.action_filter.get_filtered_action
                        )
                        if time.time() - start_t > 1:
                            self.motor_targets = filtered_motor_targets

                    self.prev_motor_targets = self.motor_targets.copy()

                    head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                    self.motor_targets[5:9] = head_motor_targets

                    action_dict = self.run_stage(
                        "loop.make_action_dict",
                        make_action_dict,
                        self.motor_targets,
                        list(self.hwi.joints.keys()),
                    )

                    self.run_stage("loop.hwi.set_position_all", self.hwi.set_position_all, action_dict)

                    i += 1

                    took = time.time() - t
                    budget = 1 / self.control_freq
                    if (budget - took) < 0:
                        self.logger.log(
                            f"OVERRUN took_s={took:.4f} budget_s={budget:.4f} excess_s={took - budget:.4f}"
                        )
                    time.sleep(max(0, budget - took))

                except Exception:
                    failure_count += 1
                    if not self.continue_on_error:
                        self.logger.log("RUN_STOP reason=exception continue_on_error=False")
                        raise
                    self.logger.log(f"LOOP_CONTINUE_AFTER_EXCEPTION failure_count={failure_count}")

        except KeyboardInterrupt:
            self.logger.log("RUN_STOP reason=keyboard_interrupt")
        finally:
            self.shutdown()

        if self.save_obs:
            self.run_stage("save_obs", pickle.dump, self.saved_obs, open("robot_saved_obs.pkl", "wb"))

        self.logger.log("RUN_END")


def parse_args():
    parser = argparse.ArgumentParser(description="Policy execution with full stage trace")
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--duck_config_path", type=str, default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--serial_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0.0, help="deg")
    parser.add_argument(
        "--commands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gamepad command path",
    )
    parser.add_argument("--save_obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay_obs", type=str, default=None)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means infinite")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument(
        "--strict_hwi_reads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use low-level read_present_* to expose serial exceptions instead of silent None",
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, keep looping after exceptions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = TraceLogger(args.logdir)
    logger.log("SCRIPT_START")

    pid = [args.p, args.i, args.d]

    runner = RLWalkTrace(
        onnx_model_path=args.onnx_model_path,
        logger=logger,
        duck_config_path=args.duck_config_path,
        serial_port=args.serial_port,
        control_freq=args.control_freq,
        pid=pid,
        action_scale=args.action_scale,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        strict_hwi_reads=args.strict_hwi_reads,
        continue_on_error=args.continue_on_error,
    )
    runner.run(max_steps=args.max_steps)

    logger.log("SCRIPT_END")


if __name__ == "__main__":
    main()