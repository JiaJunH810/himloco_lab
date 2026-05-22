#!/usr/bin/env python3
# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2Sim: Isaac Lab → MuJoCo deployment for HimLoco G1 policy."""

import os
import sys
import argparse

import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime

sys.path.insert(0, os.path.dirname(__file__))

from keyboard_control import KeyboardCommand


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ── Quaternion utilities (wxyz convention) ────────────────────────────

def quat_conjugate(q):
    """q: (..., 4) in (w, x, y, z)."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate([q[:, 0:1], -q[:, 1:4]], axis=-1).reshape(shape)


def quat_apply(q, v):
    """Apply quaternion rotation q (w,x,y,z) to vector v (x,y,z)."""
    shape = v.shape
    q = q.reshape(-1, 4).astype(np.float64)
    v = v.reshape(-1, 3).astype(np.float64)
    xyz = q[:, 1:4]
    t = np.cross(xyz, v) * 2.0
    return (v + q[:, 0:1] * t + np.cross(xyz, t)).reshape(shape).astype(np.float32)


def quat_apply_inverse(q, v):
    """Apply inverse quaternion rotation. q: (w,x,y,z), v: (x,y,z)."""
    return quat_apply(quat_conjugate(q), v)


def quat_inv(q, eps=1e-9):
    """Inverse of quaternion (w,x,y,z)."""
    return quat_conjugate(q) / (q ** 2).sum(axis=-1, keepdims=True).clip(min=eps)


def quat_mul(q1, q2):
    """Multiply two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# HimLoco G1 Sim2Sim
# ═══════════════════════════════════════════════════════════════════════

class HimLocoG1Sim2Sim:
    def __init__(self, xml_path, export_dir):
        # ── Load MuJoCo model ─────────────────────────────────────────
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        # ── Load ONNX metadata & policy ────────────────────────────────
        policy_path = os.path.join(export_dir, "policy.onnx")
        encoder_path = os.path.join(export_dir, "encoder.onnx")
        model = onnx.load(policy_path)
        self._load_metadata(model)

        self.encoder = onnxruntime.InferenceSession(encoder_path)
        self.policy = onnxruntime.InferenceSession(policy_path)

        # ── Simulation parameters ──────────────────────────────────────
        self.sim_dt = self.m.opt.timestep
        self.decimation = max(1, round(0.02 / self.sim_dt))
        self.policy_dt = self.decimation * self.sim_dt

        # ── History buffer for encoder ─────────────────────────────────
        self.history_length = 6
        self.num_one_step_obs = 45
        self.obs_history = np.zeros(self.history_length * self.num_one_step_obs,
                                     dtype=np.float32)

        # ── State ──────────────────────────────────────────────────────
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.step_counter = 0

        # ── Adjust initial height so feet touch ground ─────────────────
        self._adjust_initial_height()

        # ── Pre-fill history buffer ────────────────────────────────────
        self._prefill_history()

        # ── Viewer ─────────────────────────────────────────────────────
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer._paused = None  # 禁用空格/右键暂停，避免方向键→与键盘控制冲突
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -15

        # ── Keyboard command ──────────────────────────────────────────
        self.keyboard = KeyboardCommand(vx_step=0.5, vy_step=0.5, max_v=2.0)

        print(f"[INFO] G1 Sim2Sim initialized.")
        print(f"  qpos_joint_order: {self.qpos_joint_order}")
        print(f"  lab_order: {self.lab_order}")
        print(f"  sim_dt={self.sim_dt:.4f}, decimation={self.decimation}, "
              f"policy_dt={self.policy_dt:.4f}")
        print(f"  cmd: vx=1.0 vy=0.0 omega=0.0")

    # ── Metadata loading ───────────────────────────────────────────────

    def _load_metadata(self, model):
        """Read joint names, defaults, scales from ONNX metadata."""
        # self.m.nu includes all 29 actuators (waist + arms).
        # We only control the 12 leg joints matching the ONNX policy output.
        self.mj_nu = self.m.nu
        self.num_action = None  # set after parsing lab_order

        def _parse_list(val):
            s = val.strip()
            if s.startswith('['):
                s = s.strip('[]')
            return np.array([float(x) for x in s.split(',') if x.strip()])

        def _parse_str_list(val):
            """Parse ONNX metadata value to list of strings."""
            s = val.strip()
            if s.startswith('['):
                s = s.strip('[]')
            return [x.strip().strip("'\"") for x in s.split(',') if x.strip()]

        print("========================== lab parameters ==========================")
        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.lab_order = _parse_str_list(prop.value)
            if prop.key == "default_joint_pos":
                self.lab_default_joint_pos = _parse_list(prop.value)
            if prop.key == "joint_stiffness":
                self.lab_joint_stiffness = _parse_list(prop.value)
            if prop.key == "joint_damping":
                self.lab_joint_damping = _parse_list(prop.value)
            if prop.key == "action_scale":
                self.lab_action_scale = _parse_list(prop.value)
            if prop.key == "observation_names":
                self.lab_obs_names = _parse_str_list(prop.value)

        print(f"lab_order: {self.lab_order}")
        print(f"default_joint_pos: {self.lab_default_joint_pos}")
        print(f"joint_stiffness: {self.lab_joint_stiffness}")
        print(f"joint_damping: {self.lab_joint_damping}")
        print(f"action_scale: {self.lab_action_scale}")

        # Build qpos joint order using mujoco joint name lookups.
        # MuJoCo qpos order is body-tree traversal, NOT actuator order!
        self.qpos_joint_order = []
        self.qpos_joint_adr = []
        self.qvel_joint_adr = []
        for i in range(self.m.njnt):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.m.jnt_qposadr[i]
            dof_adr = self.m.jnt_dofadr[i]
            if qpos_adr >= 0:
                self.qpos_joint_order.append(name)
                self.qpos_joint_adr.append(qpos_adr)
                self.qvel_joint_adr.append(dof_adr)

        # Remove non-actuated joints (free joint, etc.)
        actuated_joints = set()
        for i in range(self.m.nu):
            jnt_id = self.m.actuator_trnid[i, 0]
            jnt_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            actuated_joints.add(jnt_name)

        filtered_order = []
        filtered_adr = []
        filtered_dof_adr = []
        lab_joint_set = set(self.lab_order)
        for name, adr, dof_adr in zip(self.qpos_joint_order, self.qpos_joint_adr, self.qvel_joint_adr):
            if name in actuated_joints and name in lab_joint_set:
                filtered_order.append(name)
                filtered_adr.append(adr)
                filtered_dof_adr.append(dof_adr)
        self.qpos_joint_order = filtered_order
        self.qpos_joint_adr = filtered_adr
        self.qvel_joint_adr = filtered_dof_adr
        print(f"qpos_joint_order: {self.qpos_joint_order}")

        # Build qpos_order ↔ lab_order mappings
        self.qpos_to_lab = [self.lab_order.index(j) for j in self.qpos_joint_order]
        self.lab_to_qpos = [self.qpos_joint_order.index(j) for j in self.lab_order]
        print(f"qpos_to_lab: {self.qpos_to_lab}")
        print(f"lab_to_qpos: {self.lab_to_qpos}")

        # Number of actions matches the policy output (12 leg joints)
        self.num_action = len(self.lab_order)
        print(f"num_action (lab): {self.num_action} (MJCF has {self.mj_nu} actuators)")

        # Build actuator index for each lab joint (for writing ctrl)
        # G1 MJCF motors are named with "_joint" suffix (e.g. "left_hip_pitch_joint")
        self.act_for_lab = []
        for lab_joint_name in self.lab_order:
            act_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, lab_joint_name)
            self.act_for_lab.append(act_id)
        print(f"act_for_lab: {self.act_for_lab}")

        # Default joint positions in qpos order (for initial pose)
        self.default_joint_pos_qpos = self.lab_default_joint_pos[self.qpos_to_lab]

    # ── Initialization helpers ─────────────────────────────────────────

    def _adjust_initial_height(self):
        """Set base height so feet just touch the ground."""
        self._set_joint_positions(self.default_joint_pos_qpos)
        mujoco.mj_forward(self.m, self.d)

        # Find lowest collision geom z-position (foot contact points)
        lowest = float("inf")
        for i in range(self.m.ngeom):
            if self.m.geom_contype[i] and self.m.geom_conaffinity[i]:
                lowest = min(lowest, self.d.geom_xpos[i, 2])
        print(f"[INFO] Lowest foot geom z={lowest:.3f}, "
              f"adjusted base z={self.d.qpos[2] - lowest + 0.005:.3f}")
        self.d.qpos[2] -= lowest - 0.005
        mujoco.mj_forward(self.m, self.d)

    # ── Joint state helpers (qpos order, not actuator order) ────────────

    def _set_joint_positions(self, qpos_order_values):
        """Set joint positions using qpos body-tree order."""
        for adr, val in zip(self.qpos_joint_adr, qpos_order_values):
            self.d.qpos[adr] = val

    def _get_joint_positions(self):
        """Return joint positions in qpos body-tree order."""
        pos = np.zeros(len(self.qpos_joint_adr), dtype=np.float32)
        for i, adr in enumerate(self.qpos_joint_adr):
            pos[i] = self.d.qpos[adr]
        return pos

    def _get_joint_velocities(self):
        """Return joint velocities in qpos body-tree order."""
        vel = np.zeros(len(self.qvel_joint_adr), dtype=np.float32)
        for i, adr in enumerate(self.qvel_joint_adr):
            vel[i] = self.d.qvel[adr]
        return vel

    # ── Initialization helpers ─────────────────────────────────────────

    def _prefill_history(self):
        """Fill history buffer with current observation to avoid cold start."""
        init_cmd = np.zeros(3, dtype=np.float32)
        obs = self._compute_obs(init_cmd)
        for _ in range(self.history_length):
            self.obs_history = np.concatenate(
                [obs, self.obs_history[: -self.num_one_step_obs]])
        print("[INFO] History buffer pre-filled.")

    # ── Observation computation ────────────────────────────────────────

    def _compute_obs(self, command):
        """Compute 45-dim policy observation.

        Order (from deploy config):
          velocity_commands(3)  base_ang_vel(3)  projected_gravity(3)
          joint_pos_rel(12)     joint_vel_rel(12)  last_action(12)
        """
        # Extract MuJoCo state (qpos order)
        qpos_joint_pos = self._get_joint_positions()
        qpos_joint_vel = self._get_joint_velocities()
        root_quat = self.d.qpos.astype(np.float32)[3:7]

        # velocity_commands (3): scale=1.0, clip=[-100,100]
        vel_cmd = np.clip(command.astype(np.float32), -100.0, 100.0)

        # base_ang_vel (3): body frame, scale=0.25
        base_ang_vel = self.d.qvel.astype(np.float32)[3:6]
        base_ang_vel = np.clip(base_ang_vel * 0.25, -100.0, 100.0)

        # projected_gravity (3): [0,0,-1] → body, scale=1.0
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        proj_grav = quat_apply_inverse(root_quat, gravity_w)
        proj_grav = np.clip(proj_grav, -100.0, 100.0)

        # joint_pos_rel (12): q - default, reorder to lab, scale=1.0
        joint_pos = qpos_joint_pos[self.lab_to_qpos] - self.lab_default_joint_pos
        joint_pos = np.clip(joint_pos, -100.0, 100.0)

        # joint_vel_rel (12): dq, reorder to lab, scale=0.05
        joint_vel = qpos_joint_vel[self.lab_to_qpos] * 0.05
        joint_vel = np.clip(joint_vel, -100.0, 100.0)

        # last_action (12): scale=1.0
        last_act = np.clip(self.last_action.copy(), -100.0, 100.0)

        return np.concatenate([
            vel_cmd, base_ang_vel, proj_grav,
            joint_pos, joint_vel, last_act,
        ]).astype(np.float32)

    # ── ONNX inference ─────────────────────────────────────────────────

    def _run_policy(self):
        """Run encoder → policy dual-network inference."""
        obs_history = self.obs_history.reshape(1, -1)
        encoder_out = self.encoder.run(
            ['encoder_output'], {'obs_history': obs_history})[0]  # [1, 19]
        current_obs = obs_history[:, :self.num_one_step_obs]       # [1, 45]
        policy_input = np.concatenate([current_obs, encoder_out], axis=1)  # [1, 64]
        actions = self.policy.run(
            ['actions'], {'obs': policy_input.astype(np.float32)})[0].squeeze()
        return actions

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self):
        """Run the sim2sim loop."""
        sim_duration = 120.0
        total_steps = int(sim_duration / self.sim_dt)

        print("[INFO] Starting G1 sim2sim loop...")
        for i in range(total_steps):
            qpos_joint_pos = self._get_joint_positions()
            qpos_joint_vel = self._get_joint_velocities()

            if i % self.decimation == 0:
                # ── Keyboard velocity command ───────────────────────
                cmd = self.keyboard.update(self.viewer.window)

                # ── Compute observation ───────────────────────────────
                obs = self._compute_obs(cmd)
                self.obs_history = np.concatenate(
                    [obs, self.obs_history[: -self.num_one_step_obs]])

                # ── ONNX inference ────────────────────────────────────
                lab_actions = self._run_policy()
                self.last_action = lab_actions.copy()

                # ── Decode action → PD target in lab order ────────────
                scale_actions = lab_actions * self.lab_action_scale
                pd_target_lab = scale_actions + self.lab_default_joint_pos

                # ── Viewer ────────────────────────────────────────────
                self.viewer.cam.lookat[:] = self.d.qpos[:3]
                self.viewer.render()

                # ── Logging ────────────────────────────────────────────
                self.step_counter += 1
                if self.step_counter % 50 == 0:
                    print(f"  step {self.step_counter:5d} | "
                          f"base_z={self.d.qpos[2]:.3f} | "
                          f"action_range=[{lab_actions.min():+.2f}, {lab_actions.max():+.2f}]")

            # ── Reorder joint state to lab order for PD control ───────
            joint_pos_lab = qpos_joint_pos[self.lab_to_qpos]
            joint_vel_lab = qpos_joint_vel[self.lab_to_qpos]

            # ── PD torque control (lab order → actuator order) ────────
            torque_lab = pd_control(
                pd_target_lab, joint_pos_lab,
                self.lab_joint_stiffness,
                np.zeros(self.num_action, dtype=np.float64),
                joint_vel_lab,
                self.lab_joint_damping,
            )
            self.d.ctrl[self.act_for_lab] = torque_lab
            mujoco.mj_step(self.m, self.d)

        self.viewer.close()
        print("[INFO] G1 Sim2Sim finished.")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HimLoco G1 Sim2Sim")
    parser.add_argument("--export-dir", type=str, required=True,
                        help="Path to exported model directory.")
    args = parser.parse_args()

    xml_path = os.path.join(os.path.dirname(__file__), "assets", "g1", "scene.xml")

    sim = HimLocoG1Sim2Sim(
        xml_path=xml_path,
        export_dir=args.export_dir,
    )
    sim.run()
