import numpy as np
from src.robot_kinematics import forward_kinematics_two_link, true_jacobian

def znn_update(angles, actual_pos, desired_pos, desired_velocity, tau=0.0001, h=1.0):
    """
    ZNN 更新公式，严格按照公式 (14) 实现。
    参数:
        angles (np.ndarray): 当前关节角度。
        actual_pos (np.ndarray): 当前末端执行器位置。
        desired_pos (np.ndarray): 期望末端执行器位置。
        desired_velocity (np.ndarray): 期望末端执行器速度。
        tau (float): ZNN 的时间常数。
        h (float): 误差增益。
    返回:
        np.ndarray: 更新后的关节角度。
    """
    J = true_jacobian(angles)[:2, :]   # 计算当前雅可比矩阵
    J_pseudo = np.linalg.pinv(J)  # 计算伪逆雅可比
    # 确保 pos_error 和 desired_velocity 是 numpy 数组
    pos_error = np.array(actual_pos) - np.array(desired_pos)
    desired_velocity = np.array(desired_velocity)
    correction_term = J_pseudo @ (tau * desired_velocity - h * pos_error)

    # 更新角度
    next_angles = angles + correction_term
    return next_angles

def znn_trajectory_tracking(initial_angles, trajectory, tau=0.0001, h=1.0):
    """
    基于模型的 ZNN 轨迹追踪。
    参数:
        initial_angles (np.ndarray): 初始关节角度。
        trajectory (list): 每个点包含 (期望位置, 期望速度) 的轨迹。
        tau (float): ZNN 的时间常数。
        h (float): 误差增益。
    返回:
        tuple: (关节角度历史, 追踪的末端位置历史)。
    """
    angles = np.copy(initial_angles)
    znn_positions = []
    angles_history = [] 

    for target_pos, target_vel in trajectory:
        current_pos = forward_kinematics_two_link(angles[0], angles[1])[:2]  # 解包角度，传递 theta1 和 theta2
        angles = znn_update(angles, current_pos, target_pos, target_vel, tau, h)
        znn_positions.append(current_pos)
        angles_history.append(angles.copy())

    return np.array(angles_history), np.array(znn_positions)