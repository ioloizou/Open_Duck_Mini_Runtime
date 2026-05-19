import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        debug_log_path=None,
        debug_action_jump_threshold=0.35,
        debug_gravity_norm_tolerance=0.25,
    ) -> None:

        # Loading the imu orientation and the
        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.commands = commands

        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14

        # Found through motor indefication (rad/s)
        self.max_motor_velocity = 5.24

        # Control
        self.control_freq = control_freq
        self.pid = pid
        self.action_scale = action_scale

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.debug_log_path = debug_log_path
        self.debug_action_jump_threshold = debug_action_jump_threshold
        self.debug_gravity_norm_tolerance = debug_gravity_norm_tolerance
        self._first_timing_overrun_logged = False
        self._first_action_jump_logged = False
        self._first_gravity_anomaly_logged = False
        self.debug_data = None

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(self.control_freq, cutoff_frequency)

        self.hwi = HWI(self.duck_config, serial_port)

        if self.debug_log_path is not None:
            self.debug_data = {
                "metadata": {
                    "created_at": time.time() + "Z",
                    "onnx_model_path": self.onnx_model_path,
                    "duck_config_path": duck_config_path,
                    "serial_port": serial_port,
                    "control_freq": self.control_freq,
                    "pid": self.pid,
                    "action_scale": self.action_scale,
                    "cutoff_frequency": cutoff_frequency,
                    "max_motor_velocity": self.max_motor_velocity,
                    "num_dofs": self.num_dofs,
                    "joint_names": list(self.hwi.joints.keys()),
                    "debug_action_jump_threshold": self.debug_action_jump_threshold,
                    "debug_gravity_norm_tolerance": self.debug_gravity_norm_tolerance,
                },
                "samples": [],
                "events": [],
            }

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets")
        if self.duck_config.antennas:
            self.antennas = Antennas()

    def get_obs(self):
        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"Expected {self.num_dofs} dof positions, but got {len(dof_pos)}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"Expected {self.num_dofs} dof velocities, but got {len(dof_vel)}")
            return None

        cmds = self.last_commands

        projected_gravity = None
        if "quaternion" in imu_data and imu_data["quaternion"] is not None:
            imu_quat = imu_data["quaternion"]
            if len(imu_quat) == 4 and all(q is not None for q in imu_quat):
                try:
                    projected_gravity = self.compute_projected_gravity(imu_data)
                except Exception as e:
                    print(f"Error computing projected gravity: {e}")

        if projected_gravity is None:
            # If we can't compute the projected gravity, we can use a default value
            projected_gravity = np.array([0.0, 0.0, -1.0])

        obs = np.concatenate(
            [
                imu_data["gyro"],
                #imu_data["accelero"],
                projected_gravity,
                dof_pos - self.init_pos,
                dof_vel,
                self.last_action,
                cmds[:3],
            ]
        )
        obs_components = {
            "gyro": np.array(imu_data["gyro"], dtype=float),
            "projected_gravity": np.array(projected_gravity, dtype=float),
            "dof_pos": np.array(dof_pos, dtype=float),
            "dof_vel": np.array(dof_vel, dtype=float),
            "commands": np.array(cmds, dtype=float),
            "imu_orientation": (
                np.array(imu_data["orientation"], dtype=float)
                if "orientation" in imu_data and imu_data["orientation"] is not None
                else None
            ),
            "imu_quaternion": (
                np.array(imu_data["quaternion"], dtype=float)
                if "quaternion" in imu_data and imu_data["quaternion"] is not None
                else None
            ),
        }
        return obs, obs_components

    def log_event(self, event_name, loop_t, payload=None):
        if self.debug_data is None:
            return
        event_payload = {
            "name": event_name,
            "t": float(loop_t),
        }
        if payload is not None:
            event_payload["payload"] = payload
        self.debug_data["events"].append(event_payload)

    def log_sample(
        self,
        loop_t,
        obs_components,
        obs,
        action,
        action_jump,
        action_jump_norm,
        motor_targets,
        target_jump,
        target_jump_norm,
        head_motor_targets,
        left_trigger,
        right_trigger,
        took,
        budget_margin,
        overrun,
        paused,
    ):
        if self.debug_data is None:
            return

        gravity_norm = float(np.linalg.norm(obs_components["projected_gravity"]))
        vel_ratio = np.abs(obs_components["dof_vel"]) / self.max_motor_velocity

        sample = {
            "t": float(loop_t),
            "gyro": obs_components["gyro"].tolist(),
            "projected_gravity": obs_components["projected_gravity"].tolist(),
            "projected_gravity_norm": gravity_norm,
            "imu_orientation": (
                obs_components["imu_orientation"].tolist()
                if obs_components["imu_orientation"] is not None
                else None
            ),
            "imu_quaternion": (
                obs_components["imu_quaternion"].tolist()
                if obs_components["imu_quaternion"] is not None
                else None
            ),
            "dof_pos": obs_components["dof_pos"].tolist(),
            "dof_pos_error": (obs_components["dof_pos"] - self.init_pos).tolist(),
            "dof_vel": obs_components["dof_vel"].tolist(),
            "dof_vel_saturation_ratio": vel_ratio.tolist(),
            "commands": obs_components["commands"].tolist(),
            "obs": obs.tolist(),
            "action": action.tolist(),
            "action_jump": action_jump.tolist(),
            "action_jump_norm": float(action_jump_norm),
            "motor_targets": motor_targets.tolist(),
            "motor_target_jump": target_jump.tolist(),
            "motor_target_jump_norm": float(target_jump_norm),
            "head_motor_targets": head_motor_targets.tolist(),
            "left_trigger": float(left_trigger),
            "right_trigger": float(right_trigger),
            "paused": bool(paused),
            "loop_took": float(took),
            "budget_margin": float(budget_margin),
            "overrun": bool(overrun),
            "loop_hz": float(1.0 / took) if took > 1e-9 else float("inf"),
        }
        self.debug_data["samples"].append(sample)

        if overrun and not self._first_timing_overrun_logged:
            self.log_event(
                "first_timing_overrun",
                loop_t,
                {
                    "loop_took": float(took),
                    "budget_margin": float(budget_margin),
                },
            )
            self._first_timing_overrun_logged = True

        if (
            action_jump_norm > self.debug_action_jump_threshold
            and not self._first_action_jump_logged
        ):
            self.log_event(
                "first_large_action_jump",
                loop_t,
                {
                    "threshold": float(self.debug_action_jump_threshold),
                    "action_jump_norm": float(action_jump_norm),
                },
            )
            self._first_action_jump_logged = True

        gravity_error = abs(gravity_norm - 1.0)
        if (
            gravity_error > self.debug_gravity_norm_tolerance
            and not self._first_gravity_anomaly_logged
        ):
            self.log_event(
                "first_projected_gravity_anomaly",
                loop_t,
                {
                    "tolerance": float(self.debug_gravity_norm_tolerance),
                    "projected_gravity_norm": float(gravity_norm),
                    "error_from_unit_norm": float(gravity_error),
                },
            )
            self._first_gravity_anomaly_logged = True

    def quat_apply_inverse(self, quat, vec):
        """Apply an inverse quaternion rotation to a vector."""
        w = quat[0]
        xyz = quat[1:4]

        t = np.cross(xyz, vec)
        return vec - w * t + np.cross(xyz, t)

    def compute_projected_gravity(self, imu_data):
        gravity = np.array([0, 0, -1])
        projected_gravity = self.quat_apply_inverse(imu_data["orientation"], gravity)
        return projected_gravity

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # Lower head kps
        kps[5:9] = [8] * 4

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def run(self):
        i = 0
        interrupted = False
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()
                        self.log_event("projector_toggled", time.time() - start_t)

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()
                        self.log_event("random_sound_played", time.time() - start_t)

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        self.log_event(
                            "pause_toggled",
                            time.time() - start_t,
                            {"paused": bool(self.paused)},
                        )
                        if self.paused:
                            print("Paused")
                        else:
                            print("Unpaused")

                if self.paused:
                    time.sleep(0.1)
                    continue

                prev_action = self.last_action.copy()
                prev_motor_targets = self.motor_targets.copy()

                obs_data = self.get_obs()
                if obs_data is None:
                    continue
                obs, obs_components = obs_data

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                action_jump = action - prev_action
                action_jump_norm = np.linalg.norm(action_jump)

                self.last_action = action.copy()

                self.motor_targets = self.init_pos + action * self.action_scale

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if time.time() - start_t > 1:
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()

                # To combine teleoperated head with RL policy
                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                target_jump = self.motor_targets - prev_motor_targets
                target_jump_norm = np.linalg.norm(target_jump)

                action_dict = make_action_dict(self.motor_targets, list(self.hwi.joints.keys()))

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                budget_margin = 1 / self.control_freq - took
                overrun = budget_margin < 0
                if overrun:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )

                self.log_sample(
                    loop_t=time.time() - start_t,
                    obs_components=obs_components,
                    obs=obs,
                    action=action,
                    action_jump=action_jump,
                    action_jump_norm=action_jump_norm,
                    motor_targets=self.motor_targets.copy(),
                    target_jump=target_jump,
                    target_jump_norm=target_jump_norm,
                    head_motor_targets=np.array(head_motor_targets, dtype=float),
                    left_trigger=left_trigger,
                    right_trigger=right_trigger,
                    took=took,
                    budget_margin=budget_margin,
                    overrun=overrun,
                    paused=self.paused,
                )

                self.prev_motor_targets = self.motor_targets.copy()
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            interrupted = True
            print("Keyboard interrupt, stopping control loop")

        finally:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            if self.save_obs:
                pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
            if self.debug_data is not None:
                debug_path = str(self.debug_log_path)
                self.debug_data["metadata"]["interrupted"] = interrupted
                self.debug_data["metadata"]["num_samples"] = len(self.debug_data["samples"])
                self.debug_data["metadata"]["num_events"] = len(self.debug_data["events"])
                pickle.dump(self.debug_data, open(debug_path, "wb"))
                print(f"Saved debug log to {debug_path}")
            print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--duck_config_path",
        type=str,
        default=f"{HOME_DIR}/duck_config.json",
        help="Path to the duck config json file.",
    )

    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument(
        "--debug_log_path",
        type=str,
        default=None,
        help="Optional path to save a rich debug log for plotting.",
    )
    parser.add_argument(
        "--debug_action_jump_threshold",
        type=float,
        default=0.35,
        help="Threshold used to mark the first large action jump event.",
    )
    parser.add_argument(
        "--debug_gravity_norm_tolerance",
        type=float,
        default=0.25,
        help="Tolerance on | ||projected_gravity|| - 1 | to mark gravity anomalies.",
    )

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing arguments, starting RL walk")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        debug_log_path=args.debug_log_path,
        debug_action_jump_threshold=args.debug_action_jump_threshold,
        debug_gravity_norm_tolerance=args.debug_gravity_norm_tolerance,
    )
    print("Done instantiating RLWalk")
    rl_walk.run()
