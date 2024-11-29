import numpy as np
import matplotlib.pyplot as plt

# 全局常量
DT = 0.001  # 时间步长
TD = 20  # 总时间（控制时长）
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = np.pi / 2

# 定义搜索范围
beta1_values = np.linspace(0.6, 1.0, 10)  # 在 0.1 到 5.0 之间搜索 10 个 \(\beta_1\)
errors_over_beta = []  # 记录每个 \(\beta_1\) 的误差随时间变化


# 定义两连杆的正向运动学
def forward_kinematics(angles):
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = angles
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([x1, y1]), np.array([x2, y2])  # 返回2D坐标

# 初始化雅克比矩阵
# def initialize_jacobian(angles, delta=0.001):
#     J = np.zeros((2, 2))
#     for i in range(2):
#         perturbed_angles = angles.copy()
#         perturbed_angles[i] += delta
#         _, perturbed_end = forward_kinematics(perturbed_angles)
#         _, original_end = forward_kinematics(angles)
#         J[:, i] = (perturbed_end - original_end) / delta
#     return J

def initialize_jacobian(angles):
    l1, l2 = 1.0, 1.0
    theta1, theta2 = angles
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)
    return np.array([[j11, j12], [j21, j22]])

# 计算加速度，基于欧拉差分法
def calculate_acceleration(vel_current, vel_prev, dt=DT):
    return (vel_current - vel_prev) / dt

# ZNN1 更新公式
def znn1_update(angle_dot, pos, target_pos, target_vel, jacobian, beta):
    # 计算位置误差
    pos_error = target_pos - pos
    # 分步计算
    term1 = np.eye(2) - jacobian.T @ jacobian  # 投影调整项
    term2 = jacobian.T @ target_vel           # 目标速度项
    term3 = beta * jacobian.T @ pos_error     # 误差修正项
    # 合并结果
    angle_dot = term1 @ angle_dot + term2 + term3
    return angle_dot, pos_error

# # ZNN2 更新公式
# def znn2_update(jacobian, jacobian_dot, angles_dot, angles_dotdot, actual_acceleration, current_velocity, beta=BETA_ZNN2):
#     # 分步计算
#     term1 = actual_acceleration - jacobian @ angles_dotdot  # 实际加速度项
#     term2 = beta * (current_velocity - jacobian @ angles_dot)  # 误差修正项
#     term3 = term1 + term2  # 合并前两项
#     jacobian_dot = term3 @ angles_dot.T + jacobian_dot @ (np.eye(2) - angles_dot @ angles_dot.T)
#     return jacobian_dot

# 离散更新公式
def discrete_update(current, derivative, dt=DT):
    return current + dt * derivative

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
            #target_vel = (trajectory[i] - trajectory[i - 1]) / dt
            target_vel = (trajectory[i] - trajectory[i - 1]) / 0.0001
        else:
            target_vel = np.zeros(2)  # 初始时刻速度为0
        velocities.append(target_vel)

    return trajectory, velocities

# def znn_trajectory_tracking(initial_angles, trajectory, velocities, dt):
#     angles = np.copy(initial_angles)
#     angle_dot = np.zeros(2)  # 初始关节速度
#     jacobian = initialize_jacobian(angles)  # 初始雅克比矩阵
#     jacobian_dot = np.zeros_like(jacobian)  # 初始雅克比变化率

#     znn_positions = []

#     prev_pos = None
#     prev_vel = np.zeros(2)  # 初始末端速度
#     prev_angle_dot = np.zeros(2)  # 初始关节速度

#     for k in range(len(trajectory)):  # 使用轨迹长度直接控制循环
#         # 当前目标位置和速度
#         target_pos = trajectory[k]
#         target_vel = velocities[k]

#         # 计算当前末端位置
#         _, current_pos = forward_kinematics(angles)
#         znn_positions.append(current_pos)

#         # ZNN1 计算关节角速度
#         angle_dot = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian)

#         # 离散更新关节角度
#         angles = discrete_update(angles, angle_dot, dt)

#         # 计算末端速度
#         if prev_pos is not None:
#             actual_vel = (current_pos - prev_pos) / dt
#         else:
#             actual_vel = np.zeros(2)

#         # 计算末端加速度
#         if prev_pos is not None:
#             actual_acc = calculate_acceleration(actual_vel, prev_vel, dt)
#         else:
#             actual_acc = np.zeros(2)

#         # 计算关节加速度
#         angles_dotdot = calculate_acceleration(angle_dot, prev_angle_dot, dt)

#         # ZNN2 更新雅克比变化率
#         # jacobian_dot = znn2_update(jacobian, jacobian_dot, angle_dot, angles_dotdot, actual_acc, actual_vel)

#         # # 离散更新雅克比矩阵
#         # jacobian = discrete_update(jacobian, jacobian_dot, dt)
#         jacobian = initialize_jacobian(angles)

#         # 保存上一时刻的值
#         prev_pos = current_pos
#         prev_vel = actual_vel
#         prev_angle_dot = angle_dot

#     return znn_positions
# 修改的轨迹追踪函数，用于记录误差
def znn_trajectory_tracking_with_error(beta1, trajectory, velocities, dt):
    angles = np.copy(initial_angles)
    angle_dot = np.zeros(2)  # 初始关节速度
    jacobian = initialize_jacobian(angles)  # 初始雅克比矩阵

    errors = []  # 用于记录误差随时间的变化
    for k in range(len(trajectory)):
        # 当前目标位置和速度
        target_pos = trajectory[k]
        target_vel = velocities[k]

        # 计算当前末端位置
        _, current_pos = forward_kinematics(angles)

        # ZNN1 计算关节角速度并记录误差
        angle_dot, pos_error = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian, beta1)
        errors.append(np.linalg.norm(pos_error))  # 记录误差的范数

        # 离散更新关节角度
        angles = discrete_update(angles, angle_dot, dt)

        # 重新计算雅克比矩阵
        jacobian = initialize_jacobian(angles)

    return errors


# 可视化轨迹
def plot_trajectory(target_positions, znn_positions):
    target_x, target_y = zip(*target_positions)
    znn_x, znn_y = zip(*znn_positions)

    plt.figure()
    plt.plot(target_x, target_y, 'r--', label='Target Trajectory')
    plt.plot(znn_x, znn_y, 'b-', label='ZNN Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('ZNN Trajectory Tracking')
    plt.axis('equal')
    plt.show()

# 主程序


trajectory, velocities = generate_trajectory_with_velocity(initial_angles, delta_angle, TD, DT)
# znn_positions = znn_trajectory_tracking(initial_angles, trajectory, velocities, DT)
# plot_trajectory(trajectory, znn_positions)

for beta1 in beta1_values:
    errors = znn_trajectory_tracking_with_error(beta1, trajectory, velocities, DT)
    errors_over_beta.append(errors)

# 可视化结果
plt.figure(figsize=(10, 6))
for i, beta1 in enumerate(beta1_values):
    plt.plot(errors_over_beta[i], label=f"Beta1={beta1:.2f}")
plt.xlabel("Time Steps")
plt.ylabel("Error Norm")
plt.title("Error Convergence for Different Beta1 Values")
plt.legend()
plt.grid()
plt.show()
