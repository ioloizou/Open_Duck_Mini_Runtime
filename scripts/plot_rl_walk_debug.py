import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


def to_array(samples, key, default=None, dtype=float):
    values = []
    for sample in samples:
        value = sample.get(key, default)
        if value is None:
            values.append(np.nan)
        else:
            values.append(value)
    return np.array(values, dtype=dtype)


def to_matrix(samples, key):
    rows = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            rows.append([])
        else:
            rows.append(value)

    max_len = max((len(r) for r in rows), default=0)
    if max_len == 0:
        return np.empty((len(rows), 0), dtype=float)

    matrix = np.full((len(rows), max_len), np.nan, dtype=float)
    for i, row in enumerate(rows):
        if row:
            matrix[i, : len(row)] = np.array(row, dtype=float)
    return matrix


def add_event_markers(axes, events):
    if not events:
        return

    color_map = {
        "pause_toggled": "tab:orange",
        "projector_toggled": "tab:purple",
        "random_sound_played": "tab:brown",
        "first_timing_overrun": "tab:red",
        "first_large_action_jump": "tab:green",
        "first_projected_gravity_anomaly": "tab:cyan",
    }

    for event in events:
        name = event.get("name", "event")
        t = event.get("t", None)
        if t is None:
            continue

        color = color_map.get(name, "tab:gray")
        for ax in axes:
            ax.axvline(t, color=color, linestyle="--", alpha=0.25, linewidth=1.0)


def summarize_anomalies(samples, events):
    action_jump_norm = to_array(samples, "action_jump_norm")
    gravity_norm = to_array(samples, "projected_gravity_norm")
    budget_margin = to_array(samples, "budget_margin")

    overruns = np.sum(budget_margin < 0)
    max_action_jump = np.nanmax(action_jump_norm) if action_jump_norm.size > 0 else np.nan
    gravity_err = np.abs(gravity_norm - 1.0)
    max_gravity_err = np.nanmax(gravity_err) if gravity_err.size > 0 else np.nan

    print("===== RL WALK DEBUG SUMMARY =====")
    print(f"samples: {len(samples)}")
    print(f"events: {len(events)}")
    print(f"timing overruns: {int(overruns)}")
    print(f"max action jump norm: {max_action_jump:.4f}")
    print(f"max |gravity_norm - 1|: {max_gravity_err:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot rich RL walk debug log.")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to debug pickle file.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output image path. If omitted, an interactive window is shown.",
    )
    parser.add_argument(
        "--no_scatter",
        action="store_true",
        help="Disable command-action scatter diagnostics figure.",
    )
    args = parser.parse_args()

    data = pickle.load(open(args.file, "rb"))
    metadata = data.get("metadata", {})
    samples = data.get("samples", [])
    events = data.get("events", [])

    if len(samples) == 0:
        raise RuntimeError("No samples found in debug file. Did you run with --debug_log_path?")

    t = to_array(samples, "t")
    commands = to_matrix(samples, "commands")
    gyro = to_matrix(samples, "gyro")
    projected_gravity = to_matrix(samples, "projected_gravity")
    gravity_norm = to_array(samples, "projected_gravity_norm")
    actions = to_matrix(samples, "action")
    action_jump_norm = to_array(samples, "action_jump_norm")
    motor_targets = to_matrix(samples, "motor_targets")
    head_motor_targets = to_matrix(samples, "head_motor_targets")
    target_jump_norm = to_array(samples, "motor_target_jump_norm")
    dof_pos_error = to_matrix(samples, "dof_pos_error")
    dof_vel = to_matrix(samples, "dof_vel")
    dof_vel_sat_ratio = to_matrix(samples, "dof_vel_saturation_ratio")
    loop_took = to_array(samples, "loop_took")
    budget_margin = to_array(samples, "budget_margin")

    fig, axes = plt.subplots(7, 1, figsize=(16, 22), sharex=True)

    # Row 1: external commands.
    if commands.shape[1] >= 3:
        axes[0].plot(t, commands[:, 0], label="cmd_vx")
        axes[0].plot(t, commands[:, 1], label="cmd_vy")
        axes[0].plot(t, commands[:, 2], label="cmd_yaw")
    if commands.shape[1] >= 7:
        axes[0].plot(t, commands[:, 3], linestyle=":", alpha=0.8, label="head_cmd_0")
        axes[0].plot(t, commands[:, 4], linestyle=":", alpha=0.8, label="head_cmd_1")
        axes[0].plot(t, commands[:, 5], linestyle=":", alpha=0.8, label="head_cmd_2")
        axes[0].plot(t, commands[:, 6], linestyle=":", alpha=0.8, label="head_cmd_3")
    axes[0].set_ylabel("cmd")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Row 2: gyroscope.
    if gyro.shape[1] >= 3:
        axes[1].plot(t, gyro[:, 0], label="gyro_x")
        axes[1].plot(t, gyro[:, 1], label="gyro_y")
        axes[1].plot(t, gyro[:, 2], label="gyro_z")
    axes[1].set_ylabel("gyro")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Row 3: projected gravity + norm.
    if projected_gravity.shape[1] >= 3:
        axes[2].plot(t, projected_gravity[:, 0], label="grav_x")
        axes[2].plot(t, projected_gravity[:, 1], label="grav_y")
        axes[2].plot(t, projected_gravity[:, 2], label="grav_z")
    axes[2].plot(t, gravity_norm, color="black", linewidth=1.4, label="grav_norm")
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[2].set_ylabel("gravity")
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Row 4: policy actions + action jump norm.
    for j in range(actions.shape[1]):
        axes[3].plot(t, actions[:, j], alpha=0.35, linewidth=0.8)
    ax3b = axes[3].twinx()
    ax3b.plot(t, action_jump_norm, color="black", linewidth=1.4, label="||da||")
    ax3b.set_ylabel("||da||")
    axes[3].set_ylabel("action")
    axes[3].grid(True, alpha=0.3)

    # Row 5: motor targets + head targets + jump norm.
    for j in range(motor_targets.shape[1]):
        axes[4].plot(t, motor_targets[:, j], alpha=0.22, linewidth=0.8)
    if head_motor_targets.shape[1] > 0:
        for j in range(head_motor_targets.shape[1]):
            axes[4].plot(
                t,
                head_motor_targets[:, j],
                linewidth=1.3,
                label=f"head_target_{j}",
            )
    ax4b = axes[4].twinx()
    ax4b.plot(t, target_jump_norm, color="black", linewidth=1.4, label="||dtarget||")
    ax4b.set_ylabel("||dtarget||")
    axes[4].set_ylabel("targets")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(True, alpha=0.3)

    # Row 6: joint position errors and velocity saturation ratio.
    for j in range(dof_pos_error.shape[1]):
        axes[5].plot(t, dof_pos_error[:, j], alpha=0.22, linewidth=0.8)
    ax5b = axes[5].twinx()
    if dof_vel_sat_ratio.shape[1] > 0:
        ax5b.plot(t, np.nanmax(dof_vel_sat_ratio, axis=1), color="red", linewidth=1.2, label="max|vel|/vel_max")
        ax5b.axhline(1.0, color="red", linestyle="--", alpha=0.6)
    axes[5].set_ylabel("pos_err")
    ax5b.set_ylabel("vel_sat")
    axes[5].grid(True, alpha=0.3)

    # Row 7: control timing.
    axes[6].plot(t, loop_took, label="loop_took")
    axes[6].plot(t, budget_margin, label="budget_margin")
    if "control_freq" in metadata:
        axes[6].axhline(1.0 / metadata["control_freq"], color="black", linestyle="--", alpha=0.6, label="budget")
    axes[6].axhline(0.0, color="red", linestyle="--", alpha=0.6)
    axes[6].set_ylabel("seconds")
    axes[6].set_xlabel("time [s]")
    axes[6].legend(loc="upper right")
    axes[6].grid(True, alpha=0.3)

    add_event_markers(axes, events)

    model_name = metadata.get("onnx_model_path", "unknown_model")
    fig.suptitle(f"RL Walk Debug Dashboard | model={model_name}", fontsize=14)
    plt.tight_layout()

    if args.save is not None:
        plt.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    if not args.no_scatter and commands.shape[1] >= 3 and actions.shape[1] > 0:
        scatter_fig, scatter_axes = plt.subplots(1, 3, figsize=(15, 4))
        scatter_axes[0].scatter(commands[:, 0], actions[:, 0], s=6, alpha=0.35)
        scatter_axes[0].set_xlabel("cmd_vx")
        scatter_axes[0].set_ylabel("action_0")
        scatter_axes[0].grid(True, alpha=0.3)

        y_idx = 1 if actions.shape[1] > 1 else 0
        scatter_axes[1].scatter(commands[:, 1], actions[:, y_idx], s=6, alpha=0.35)
        scatter_axes[1].set_xlabel("cmd_vy")
        scatter_axes[1].set_ylabel(f"action_{y_idx}")
        scatter_axes[1].grid(True, alpha=0.3)

        yaw_idx = 2 if actions.shape[1] > 2 else 0
        scatter_axes[2].scatter(commands[:, 2], actions[:, yaw_idx], s=6, alpha=0.35)
        scatter_axes[2].set_xlabel("cmd_yaw")
        scatter_axes[2].set_ylabel(f"action_{yaw_idx}")
        scatter_axes[2].grid(True, alpha=0.3)

        scatter_fig.suptitle("Command-Action Diagnostic Scatter")
        plt.tight_layout()
        plt.show()

    summarize_anomalies(samples, events)


if __name__ == "__main__":
    main()
