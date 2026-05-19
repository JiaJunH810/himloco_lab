#!/usr/bin/env python3
# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script: hold robot at default_joint_pos to verify joint mapping + PD + physics."""

import os
import sys
import argparse

import mujoco
import mujoco_viewer
import numpy as np
import onnx


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


class DefaultPoseTester:
    """Load MuJoCo model + ONNX metadata, hold default_joint_pos via PD control."""

    def __init__(self, xml_path, export_dir):
        # ── Load MuJoCo model ─────────────────────────────────────────
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        # ── Load ONNX metadata only (no inference) ────────────────────
        policy_path = os.path.join(export_dir, "policy.onnx")
        model = onnx.load(policy_path)
        self._load_metadata(model)

        # ── Simulation parameters ──────────────────────────────────────
        self.sim_dt = self.m.opt.timestep
        self.decimation = max(1, round(0.02 / self.sim_dt))
        self.policy_dt = self.decimation * self.sim_dt
        self.step_counter = 0

        # ── Set initial pose and adjust height ─────────────────────────
        self._set_default_pose()
        self._adjust_initial_height()

        # ── PD target: hold at default_joint_pos (in xml order) ────────
        self.pd_target = self.lab_default_joint_pos[self.lab_to_xml].copy()

        # ── Viewer ─────────────────────────────────────────────────────
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20

        print(f"[INFO] DefaultPoseTester initialized.")
        print(f"  xml_order: {self.xml_order}")
        print(f"  lab_order: {self.lab_order}")
        print(f"  lab_default_joint_pos: {self.lab_default_joint_pos}")
        print(f"  pd_target (xml order):  {self.pd_target}")
        print(f"  xml_to_lab: {self.xml_to_lab}")
        print(f"  lab_to_xml: {self.lab_to_xml}")
        print(f"  joint_stiffness: {self.lab_joint_stiffness}")
        print(f"  joint_damping:   {self.lab_joint_damping}")
        print(f"  sim_dt={self.sim_dt:.4f}, decimation={self.decimation}, "
              f"policy_dt={self.policy_dt:.4f}")

    # ── Metadata loading ───────────────────────────────────────────────

    def _load_metadata(self, model):
        """Read joint names, defaults, stiffness, damping from ONNX metadata."""
        print("========================== xml parameters ==========================")
        self.xml_order = []
        for i in range(self.m.nu):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.xml_order.append(name)
        self.num_action = len(self.xml_order)
        self.xml_order_joint = [n + '_joint' for n in self.xml_order]
        print(f"xml_order (actuator): {self.xml_order}")
        print(f"xml_order (joint):    {self.xml_order_joint}")
        print(f"num_action: {self.num_action}")

        def _parse_list(val):
            s = val.strip()
            if s.startswith('['):
                s = s.strip('[]')
            return np.array([float(x) for x in s.split(',') if x.strip()])

        def _parse_str_list(val):
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

        print(f"lab_order: {self.lab_order}")
        print(f"default_joint_pos: {self.lab_default_joint_pos}")
        print(f"joint_stiffness: {self.lab_joint_stiffness}")
        print(f"joint_damping:   {self.lab_joint_damping}")

        self.xml_to_lab = [self.xml_order_joint.index(j) for j in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(j) for j in self.xml_order_joint]
        print(f"xml_to_lab: {self.xml_to_lab}")
        print(f"lab_to_xml: {self.lab_to_xml}")

    # ── Initialization helpers ─────────────────────────────────────────

    def _set_default_pose(self):
        """Set joint positions to default pose."""
        xml_joint_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
        xml_joint_pos[:] = self.lab_default_joint_pos[self.lab_to_xml]
        self.d.qpos[-self.num_action:] = xml_joint_pos
        mujoco.mj_forward(self.m, self.d)

    def _adjust_initial_height(self):
        """Set base height so feet just touch the ground."""
        self._set_default_pose()

        foot_names = ["FL", "FR", "RL", "RR"]
        lowest = float("inf")
        for name in foot_names:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)
            z = self.d.geom_xpos[gid, 2] - 0.022  # sphere radius
            lowest = min(lowest, z)
        self.d.qpos[2] -= lowest - 0.005
        mujoco.mj_forward(self.m, self.d)
        print(f"[INFO] Lowest foot z={lowest:.3f}, adjusted base z={self.d.qpos[2]:.3f}")

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self):
        """Run PD control loop holding default_joint_pos."""
        sim_duration = 30.0
        total_steps = int(sim_duration / self.sim_dt)

        print("[INFO] Starting default-pose test loop...")
        print("[INFO] Robot will hold default_joint_pos via PD control.")
        print("[INFO] Observe: robot should stand stably. If it collapses or flips,")
        print("[INFO] there is an issue with joint mapping, PD gains, or the model.")

        for i in range(total_steps):
            xml_joint_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
            xml_joint_vel = self.d.qvel.astype(np.float32)[-self.num_action:]

            # ── Viewer update (every decimation step) ──────────────────
            if i % self.decimation == 0:
                self.viewer.cam.lookat[:] = self.d.qpos[:3]
                self.viewer.render()

                self.step_counter += 1
                if self.step_counter % 50 == 0:
                    pos_error = np.abs(self.pd_target - xml_joint_pos)
                    print(f"  step {self.step_counter:5d} | "
                          f"base_z={self.d.qpos[2]:.4f} | "
                          f"max_pos_error={pos_error.max():.4f} | "
                          f"mean_pos_error={pos_error.mean():.4f}")

            # ── PD torque control ─────────────────────────────────────
            torque = pd_control(
                self.pd_target, xml_joint_pos,
                self.lab_joint_stiffness[self.lab_to_xml],
                np.zeros_like(self.lab_joint_damping),
                xml_joint_vel,
                self.lab_joint_damping[self.lab_to_xml],
            )
            self.d.ctrl = torque
            mujoco.mj_step(self.m, self.d)

        self.viewer.close()
        print("[INFO] DefaultPoseTester finished.")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test: hold default_joint_pos")
    parser.add_argument("--export-dir", type=str, required=True,
                        help="Path to exported model directory.")
    args = parser.parse_args()

    xml_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")

    tester = DefaultPoseTester(
        xml_path=xml_path,
        export_dir=args.export_dir,
    )
    tester.run()
