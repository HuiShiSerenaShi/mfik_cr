import numpy as np
import matplotlib.pyplot as plt

# 全局常量
DT = 0.001  # 时间步长
BETA_ZNN1 = 20  # ZNN1 的 beta 参数
BETA_ZNN2 = 20  # ZNN2 的 beta 参数
TD = 20  # 总时间（控制时长）

# 定义两连杆的正向运动学
def forward_kinematics(angles):
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = angles
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([x1, y1, 0]), np.array([x2, y2, 0])  # 返回三维坐标，Z = 0

# 初始化雅克比矩阵
def initialize_jacobian(angles, delta=0.001):
    J = np.zeros((2, 2))
    for i in range(2):
        perturbed_angles = angles.copy()
        perturbed_angles[i] += delta
        _, perturbed_end = forward_kinematics(perturbed_angles)
        _, original_end = forward_kinematics(angles)
        J[:, i] = (perturbed_end[:2] - original_end[:2]) / delta
    return J

# 计算加速度，基于欧拉差分法
def calculate_acceleration(vel_current, vel_prev, dt=DT):
    """
    根据欧拉差分法计算加速度
    vel_current: 当前速度 (k 时刻)
    vel_prev: 上一时刻的速度 (k-1 时刻)
    dt: 时间步长
    返回：加速度 (k 时刻)
    """
    return (vel_current - vel_prev) / dt


# ZNN1 更新公式
# def znn1_update(angle_dot_k, pos_k, target_pos_k, target_vel_k, jacobian_matrix_k, beta=BETA_ZNN1):
#     # 计算位置误差
#     pos_error_k = target_pos_k[:2] - pos_k[:2]
#     # 计算当前时刻的角速度
#     angle_dot_k = (np.eye(2) - jacobian_matrix_k.T @ jacobian_matrix_k) @ angle_dot_k + jacobian_matrix_k.T @ (target_vel_k[:2] + beta * pos_error_k)
#     return angle_dot_k

def znn1_update(angle_dot_k, pos_k, target_pos_k, target_vel_k, jacobian_matrix_k, beta=BETA_ZNN1):
    # 计算位置误差
    pos_error_k = target_pos_k[:2] - pos_k[:2]

    # 分步计算
    term1 = np.eye(2) - jacobian_matrix_k.T @ jacobian_matrix_k  # 第一项
    term2 = jacobian_matrix_k.T @ target_vel_k[:2]               # 第二项（目标速度项）
    term3 = beta * jacobian_matrix_k.T @ pos_error_k             # 第三项（误差修正项）

    # 合并结果
    angle_dot_k = term1 @ angle_dot_k + term2 + term3
    return angle_dot_k



# def znn2_update(jacobian_k, jacobian_dot_k,angles_dot_k, angles_dotdot_k, actual_acceleration_k, current_velocity_k, beta=BETA_ZNN2):
#     jacobian_dot_k = (
#         (actual_acceleration_k[:2] - jacobian_k @ angles_dotdot_k[:2] + beta * (current_velocity_k[:2] - jacobian_k @ angles_dot_k[:2]))
#         @ angles_dot_k[:2].T
#         + jacobian_dot_k @ (np.eye(2) - angles_dot_k[:2] @ angles_dot_k[:2].T)
#     )
#     return jacobian_dot_k

def znn2_update(jacobian_k, jacobian_dot_k, angles_dot_k, angles_dotdot_k, actual_acceleration_k, current_velocity_k, beta=BETA_ZNN2):
    # 分步计算
    term1 = actual_acceleration_k[:2] - jacobian_k @ angles_dotdot_k[:2]  # 第一项（实际加速度 - 雅克比项）
    term2 = beta * (current_velocity_k[:2] - jacobian_k @ angles_dot_k[:2])  # 第二项（误差修正项）
    term3 = term1 + term2  # 合并前两项
    term4 = term3 @ angles_dot_k[:2].T  # 外积构造变化率

    term5 = jacobian_dot_k @ (np.eye(2) - angles_dot_k[:2] @ angles_dot_k[:2].T)

    # 合并最终结果
    jacobian_dot_k = term4 + term5
    return jacobian_dot_k



# 离散更新公式
def discrete_update(current, derivative, dt=DT):
    return current + dt * derivative

# 生成目标轨迹
# 生成目标轨迹和速度
def generate_trajectory_with_velocity(initial_angles, delta_angle, total_time, dt):
    num_points = int(total_time / dt)
    trajectory = []
    velocities = []
    angles = np.linspace(0, delta_angle, num_points)

    for i, angle in enumerate(angles):
        theta1 = initial_angles[0] + angle
        theta2 = initial_angles[1] - angle
        _, end_effector_pos = forward_kinematics([theta1, theta2])
        trajectory.append(end_effector_pos)

        # 计算目标速度
        if i > 0:
            target_vel = (trajectory[i][:2] - trajectory[i - 1][:2]) / dt
        else:
            target_vel = np.zeros(2)  # 初始时刻速度为 0
        velocities.append(target_vel)

    return trajectory, velocities

def znn_trajectory_tracking(initial_angles, trajectory, velocities, total_time, dt):
    angles = np.copy(initial_angles)
    angle_dot = np.zeros(2)  # 初始关节速度
    jacobian = initialize_jacobian(angles)  # 初始雅克比矩阵
    jacobian_dot = np.zeros_like(jacobian)  # 初始雅克比变化率

    znn_positions = []

    prev_pos = None
    prev_vel = np.zeros(2)  # 初始末端速度
    prev_angle_dot = np.zeros(2)  # 初始关节速度

    for k in range(int(total_time / dt)):
        # 当前目标位置和速度
        target_pos = trajectory[min(k, len(trajectory) - 1)]
        target_vel = velocities[min(k, len(velocities) - 1)]

        # 3.1 计算实际末端位置
        _, current_pos = forward_kinematics(angles)
        znn_positions.append(current_pos)

        # 3.2 ZNN1 计算关节角速度
        angle_dot = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian)

        # 3.3 离散更新关节角度
        angles = discrete_update(angles, angle_dot, dt)

        # 3.4 计算末端速度
        if prev_pos is not None:
            actual_vel = (current_pos[:2] - prev_pos[:2]) / dt
        else:
            actual_vel = np.zeros(2)

        # 3.5 计算末端加速度
        if prev_pos is not None:
            actual_acc = calculate_acceleration(actual_vel, prev_vel, dt)
        else:
            actual_acc = np.zeros(2)

        # 3.6 计算关节加速度
        angles_dotdot = calculate_acceleration(angle_dot, prev_angle_dot, dt)

        # 3.7 更新雅克比变化率（ZNN2）
        jacobian_dot = znn2_update(jacobian, jacobian_dot, angle_dot, angles_dotdot, actual_acc, actual_vel)

        # 3.8 离散更新雅克比矩阵
        jacobian = discrete_update(jacobian, jacobian_dot, dt)

        # 保存上一时刻的值
        prev_pos = current_pos
        prev_vel = actual_vel
        prev_angle_dot = angle_dot

    return znn_positions

# 可视化轨迹
def plot_trajectory(target_positions, znn_positions):
    target_x, target_y, _ = zip(*target_positions)
    znn_x, znn_y, _ = zip(*znn_positions)

    plt.figure()
    plt.plot(target_x, target_y, 'r--', label='Target Trajectory')
    plt.plot(znn_x, znn_y, 'b-', label='ZNN Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('ZNN Trajectory Tracking')
    plt.show()

# 主程序
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = np.pi

trajectory, velocities = generate_trajectory_with_velocity(initial_angles, delta_angle, TD, DT)
znn_positions = znn_trajectory_tracking(initial_angles, trajectory, velocities, TD, DT)
plot_trajectory(trajectory, znn_positions)
