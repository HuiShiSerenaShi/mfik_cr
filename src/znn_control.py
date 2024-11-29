import numpy as np

def znn1_update(angle_dot, pos, target_pos, target_vel, jacobian, beta):
    # 计算位置误差
    pos_error = target_pos - pos
    # 分步计算
    H_mat = np.eye(2) - jacobian.T @ jacobian  
    W_mat = jacobian.T        
    term_ = target_vel + beta * pos_error   
    # 合并结果
    angle_dot = H_mat @ angle_dot + W_mat @ term_
    return angle_dot

# ZNN2 更新公式
def znn2_update(jacobian, jacobian_dot, angles_dot, angles_dotdot, actual_acceleration, current_velocity, beta):
    # 分步计算
    term1 = actual_acceleration - jacobian @ angles_dotdot  
    term2 = beta * (current_velocity - jacobian @ angles_dot)  
    term3 = term1 + term2  # 合并前两项
    G_mat = np.eye(2) - angles_dot @ angles_dot.T
    jacobian_dot = term3 @ angles_dot.T + jacobian_dot @ G_mat
    return jacobian_dot

# 离散更新公式
def discrete_update(current, derivative, dt):
    return current + dt * derivative

