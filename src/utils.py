import numpy as np

def discrete_update(current, derivative, dt):
    """
    离散时间更新公式。
    
    参数:
        current (np.ndarray): 当前状态变量。
        derivative (np.ndarray): 导数（变化率）。
        dt (float): 时间步长。
    
    返回:
        np.ndarray: 更新后的状态变量。
    """
    return current + dt * derivative


def calculate_acceleration(current_velocity, previous_velocity, dt):
    """
    计算加速度。
    
    参数:
        current_velocity (np.ndarray): 当前速度。
        previous_velocity (np.ndarray): 之前的速度。
        dt (float): 时间步长。
    
    返回:
        np.ndarray: 加速度。
    """
    return (current_velocity - previous_velocity) / dt


def clip_angles(angles):
    """
    将角度值限制在 [0, 2π] 范围内。
    
    参数:
        angles (np.ndarray): 输入的角度数组（单位：弧度）。
    
    返回:
        np.ndarray: 限制后的角度数组。
    """
    return np.mod(angles, 2 * np.pi)


def calculate_error(target, actual):
    """
    计算误差的模。
    
    参数:
        target (np.ndarray): 目标值。
        actual (np.ndarray): 实际值。
    
    返回:
        float: 误差模（欧几里得距离）。
    """
    return np.linalg.norm(target - actual)
