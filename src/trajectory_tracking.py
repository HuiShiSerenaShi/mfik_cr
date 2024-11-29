import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # 加入根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))  # 加入 src 目录
import numpy as np
from src.robot_kinematics import forward_kinematics_two_link, true_jacobian
from src.znn_control import znn1_update, znn2_update
from src.utils import discrete_update, calculate_acceleration

def znn_trajectory_tracking(initial_angles, trajectory, velocities, dt, l1=1.0, l2=1.0, beta1=1.0, beta2=1.0):
    """
    使用 ZNN 算法进行轨迹追踪。
    
    参数:
        initial_angles (list): 初始关节角度。
        trajectory (list): 目标轨迹点列表。
        velocities (list): 目标速度列表。
        dt (float): 时间步长。
        l1, l2 (float): 连杆长度。
        beta (float): ZNN 参数。
    
    返回:
        list: 实际轨迹点。
        list: 每一时刻的关节角度。
        list: 每一时刻的误差。
    """
    angles = np.copy(initial_angles)
    angle_dot = np.zeros(2)
# 提取二维平面的线速度部分
    jacobian = true_jacobian(angles)[:2, :]
    jacobian_dot = np.zeros_like(jacobian)

    actual_positions = []
    joint_angles = []
    errors = []

    prev_pos = None
    prev_vel = np.zeros(2)
    prev_angle_dot = np.zeros(2)

    for k in range(len(trajectory)):
        target_pos = trajectory[k]
        target_vel = velocities[k]

        current_pos = forward_kinematics_two_link(angles[0], angles[1], l1, l2)[:2]
        actual_positions.append(current_pos)
        joint_angles.append(angles)

        error = np.linalg.norm(target_pos - current_pos)
        errors.append(error)

        angle_dot = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian, beta1)
        angles = discrete_update(angles, angle_dot, dt)

        if prev_pos is not None:
            actual_vel = (current_pos - prev_pos) / dt
        else:
            actual_vel = np.zeros(2)

        if prev_pos is not None:
            actual_acc = calculate_acceleration(actual_vel, prev_vel, dt)
        else:
            actual_acc = np.zeros(2)

        angles_dotdot = calculate_acceleration(angle_dot, prev_angle_dot, dt)
        # jacobian_dot = znn2_update(jacobian, jacobian_dot, angle_dot, angles_dotdot, actual_acc, actual_vel, beta2)
        # jacobian = discrete_update(jacobian, jacobian_dot, dt)
        jacobian = true_jacobian(angles)[:2, :]

        prev_pos = current_pos
        prev_vel = actual_vel
        prev_angle_dot = angle_dot

    return actual_positions, joint_angles, errors
