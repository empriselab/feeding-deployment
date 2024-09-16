# Author: Jimmy Wu, adapted by Rajat Kumar Jenamani for Emprise Lab
# Hack: fix one of the joints of 7DoF to become 6DoF
# References:
# - https://github.com/empriselab/gen3_compliant_controllers/blob/main/src/JointSpaceCompliantController.cpp
# - https://github.com/empriselab/gen3_compliant_controllers/blob/main/media/controller_formulation.pdf

import math
import queue
import threading
import time
from scipy.spatial.transform import Rotation as R

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.y = initial_value

    def filter(self, x):
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class CompliantController:
    def __init__(self, command_queue, control_type, fix_joint_hack):
        
        self.fix_joint_hack = fix_joint_hack

        if control_type not in ["joint", "task"]:
            raise ValueError(f"Invalid control type: {control_type}")
        self.control_type = control_type
        self.set_contants()

        self.q_s = None
        self.q_d = None
        self.dq_d = None
        self.q_n = None
        self.dq_n = None
        self.tau_filter = None
        self.x_s = None
        self.x_d = None
        self.gripper_pos = None
        self.command_queue = command_queue

        # OTG
        self.last_command_time = None
        self.otg = None
        self.otg_inp = None
        self.otg_out = None
        self.otg_res = None

        # self.data = []

    def set_contants(self):

        # from constants import self.POLICY_CONTROL_PERIOD
        self.POLICY_CONTROL_PERIOD = 0.1
        self.ALPHA = 0.01
        self.DT = 0.001
        self.DAMPING_FACTOR = 0.01

        if not self.fix_joint_hack:
            self.K_r = np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18])
            self.K_l = np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0])
            self.K_lp = np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
            self.K_r_inv = np.linalg.inv(self.K_r)
            self.K_r_K_l = self.K_r @ self.K_l

            if self.control_type == "joint":
                # self.K_p = np.diag([20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0])
                self.K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
                # self.K_d = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
                self.K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
            elif self.control_type == "task":
                raise NotImplementedError
        else:
            self.K_r = np.diag([0.4, 0.4, 0.4, 0.4, 0.2, 0.2])
            self.K_l = np.diag([160.0, 160.0, 160.0, 160.0, 100.0, 100.0])
            self.K_lp = np.diag([10.0, 10.0, 10.0, 10.0, 7.5, 7.5])
            self.K_r_inv = np.linalg.inv(self.K_r)
            self.K_r_K_l = self.K_r @ self.K_l
            
            if self.control_type == "joint":
                self.K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0])
                self.K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0])
            elif self.control_type == "task":
                self.K_T_p = np.diag([100.0, 100.0, 100.0, 400.0, 400.0, 400.0])
                self.K_T_d = np.diag([20, 20, 20, 40, 40, 40])

    def control_callback(self, arm):
        # Initialize variables on first call
        if self.q_s is None:
            self.q_s = arm.q.copy()
            self.q_d = arm.q.copy()
            self.dq_d = np.zeros_like(arm.q)
            self.q_n = arm.q.copy()
            self.dq_n = arm.dq.copy()
            self.tau_filter = LowPassFilter(self.ALPHA, arm.tau.copy())
            self.x_s = arm.x.copy()
            self.x_d = arm.x.copy()
            self.gripper_pos = arm.gripper_pos

            # Initialize OTG
            self.last_command_time = time.time()
            if self.control_type == "joint":
                self.otg = Ruckig(arm.n_compliant_dofs, self.DT)
                self.otg_inp = InputParameter(arm.n_compliant_dofs)
                self.otg_out = OutputParameter(arm.n_compliant_dofs)
                self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
                self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [
                    math.radians(450)
                ]
                self.otg_inp.current_position = arm.q.copy()
                self.otg_inp.current_velocity = arm.dq.copy()
                self.otg_inp.target_position = arm.q.copy()
                self.otg_inp.target_velocity = np.zeros(arm.n_compliant_dofs)
                self.otg_res = Result.Finished

        # Sensor readings
        self.q_s = (
            self.q_s + np.mod(arm.q - self.q_s + np.pi, 2 * np.pi) - np.pi
        )  # Unwrapped joint angle
        dq_s = arm.dq.copy()
        tau_s = arm.tau.copy()
        tau_s_f = self.tau_filter.filter(tau_s)

        # Check for new command
        if not self.command_queue.empty():
            if self.control_type == "joint":
                qpos, self.gripper_pos = self.command_queue.get()
                self.last_command_time = time.time()
                qpos = (
                    self.q_s + np.mod(qpos - self.q_s + np.pi, 2 * np.pi) - np.pi
                )  # Unwrapped joint angle
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working
            elif self.control_type == "task":
                x, self.gripper_pos = self.command_queue.get()
                self.last_command_time = time.time()
                self.x_d = x

        if self.control_type == "joint":
            # Maintain current pose if command stream is disrupted
            if time.time() - self.last_command_time > 2.5 * self.POLICY_CONTROL_PERIOD:
                self.otg_inp.target_position = self.otg_out.new_position
                self.otg_res = Result.Working

            # Update OTG
            if self.otg_res == Result.Working:
                self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
                self.otg_out.pass_to_input(self.otg_inp)
                self.q_d[:] = self.otg_out.new_position
                self.dq_d[:] = self.otg_out.new_velocity

        # self.data.append({
        #     'timestamp': time.time(),
        #     'q_s': self.q_s.tolist(),
        #     'dq_s': dq_s.tolist(),
        #     'q_d': self.q_d.tolist(),
        #     'dq_d': self.dq_d.tolist(),
        #     'target_position': self.otg_inp.target_position,
        #     'target_velocity': self.otg_inp.target_velocity,
        #     'new_position': self.otg_out.new_position,
        #     'new_velocity': self.otg_out.new_velocity,
        # })

        # Compute joint torque for task
        g = arm.gravity()

        # if not np.allclose(self.q_s, self.q_d, atol=1e-3):
        #     self.data.append({
        #         'q_s': self.q_s.tolist(),
        #         'q_d': self.q_d.tolist(),
        #         'q_n': self.q_n.tolist(),
        #     })

        if self.control_type == "joint":
            
            tau_task = -self.K_p @ (self.q_n - self.q_d) - self.K_d @ (self.dq_n - self.dq_d) + g

            # Nominal motor plant
            ddq_n = self.K_r_inv @ (tau_task - tau_s_f)
            self.dq_n += ddq_n * self.DT
            self.q_n += self.dq_n * self.DT

            # Nominal friction
            tau_f = self.K_r_K_l @ ((self.dq_n - dq_s) + self.K_lp @ (self.q_n - self.q_s))

            # Torque command
            tau_c = tau_task + tau_f

        elif self.control_type == "task":
            if not self.fix_joint_hack:
                raise NotImplementedError
            else:
                x_n, J_n = arm.get_fk(self.q_n)

                pos_error = x_n[:3] - self.x_d[:3]

                # Convert to Rotation objects
                R_n = R.from_quat(x_n[3:])
                R_d = R.from_quat(self.x_d[3:])

                # Adjust quaternions to be on the same hemisphere
                if np.dot(R_d.as_quat(), R_n.as_quat()) < 0.0:
                    R_n = R.from_quat(-R_n.as_quat())

                # Compute error rotation
                error_rotation = R_n.inv() * R_d

                # Convert error rotation to quaternion
                error_quat = error_rotation.as_quat()

                # Extract vector part
                orient_error_vector = error_quat[:3]

                # Get rotation matrix of nominal pose
                R_n_matrix = R_n.as_matrix()

                # Compute orientation error
                orient_error = -R_n_matrix @ orient_error_vector

                # Assemble error
                error = np.zeros(6)
                error[:3] = pos_error
                error[3:] = orient_error 

                damping_lambda = self.DAMPING_FACTOR * np.eye(arm.n_compliant_dofs)
                J_n_damped = np.linalg.inv(J_n.T @ J_n + damping_lambda) @ J_n.T

                tau_task = J_n_damped @ (-self.K_T_p @ error - self.K_T_d @ (J_n @ self.dq_n)) + g

                # Nominal motor plant
                ddq_n = self.K_r_inv @ (tau_task - tau_s_f)
                self.dq_n += ddq_n * self.DT
                self.q_n += self.dq_n * self.DT

                # Nominal friction
                tau_f = self.K_r_K_l @ ((self.dq_n - dq_s) + self.K_lp @ (self.q_n - self.q_s))

            # Torque command
            tau_c = tau_task + tau_f

        return tau_c, self.gripper_pos

# def command_loop_retract(command_queue, stop_event):
#     # qpos = np.array([-2.771089155364116, -1.4597435746030278, -1.9011992769067048, -1.0872040897239863, 0.39878180820749237, -0.8243154690389938, 2.672235278861465])
#     # qpos = np.array([-2.7611776687351686, -1.1867898028941912, -1.7014195845733209, -1.8118651360366513, 0.2697381378506211, -0.09092617856970353, 2.4944202739346184])
#     qpos = np.array(
#         [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633]
#     )  # Home
#     # qpos = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
#     gripper_pos = 0
#     while not stop_event.is_set():
#         command_queue.put((qpos, gripper_pos))
#         time.sleep(self.POLICY_CONTROL_PERIOD)


# def command_loop_circle(arm, command_queue, stop_event):
#     from ik_solver import IKSolver

#     ik_solver = IKSolver(ee_offset=0.12)
#     quat = np.array([0.707, 0.707, 0.0, 0.0])  # (x, y, z, w)
#     radius = 0.1
#     num_points = 30
#     center = np.array([0.45, 0.0, 0.2])
#     t = np.linspace(0, 2 * np.pi, num_points)
#     x = radius * np.cos(t)
#     y = radius * np.sin(t)
#     z = np.zeros(num_points)
#     points = np.column_stack((x, y, z))
#     points += center
#     gripper_pos = 0
#     while not stop_event.is_set():
#         for pos in points:
#             qpos = ik_solver.solve(pos, quat, arm.q)
#             command_queue.put((qpos, gripper_pos))
#             time.sleep(self.POLICY_CONTROL_PERIOD)
