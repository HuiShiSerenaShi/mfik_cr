import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义两连杆的正向运动学
def forward_kinematics(angles):
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = angles
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([x1, y1, 0]), np.array([x2, y2, 0])  # 返回三维坐标，Z = 0

# IENT-ZNN1 更新公式
def znn1_update(angle_k, angle_dot_k, pos_k, target_pos_k, target_vel_k, jacobian_matrix_k, beta=0.01, dt=0.01):
    pos_error_k = target_pos_k[:2] - pos_k[:2]
    angle_dot_k_next = (np.eye(2) - jacobian_matrix_k.T @ jacobian_matrix_k) @ angle_dot_k + jacobian_matrix_k.T @ (target_vel_k[:2] + beta * pos_error_k)
    angle_k_next = angle_k + dt * angle_dot_k_next  # 使用欧拉前向方法离散化更新角度
    return angle_k_next, angle_dot_k_next

# IENT-ZNN2 更新公式，用于估计雅克比矩阵
def znn2_update(jacobian_k, angles_dot_k, angles_dotdot_k, actual_acceleration_k, current_velocity_k, beta=0.01, dt=0.01):
    jacobian_dot_k = (
        (actual_acceleration_k[:2] - jacobian_k @ angles_dotdot_k[:2] + beta * (current_velocity_k[:2] - jacobian_k @ angles_dot_k[:2]))
        @ angles_dot_k[:2].T
        + jacobian_k @ (np.eye(2) - angles_dot_k[:2] @ angles_dot_k[:2].T)
    )
    jacobian_k_next = jacobian_k + dt * jacobian_dot_k
    return jacobian_k_next, jacobian_dot_k

# 生成可达的弧形轨迹
def generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt):
    trajectory = []
    angles = np.linspace(0, delta_angle, num_points)
    
    for angle in angles:
        theta1 = initial_angles[0] + angle
        theta2 = initial_angles[1] - angle
        _, end_effector_pos = forward_kinematics([theta1, theta2])
        
        if len(trajectory) > 1:
            prev_pos = trajectory[-1][0]
            prev_prev_pos = trajectory[-2][0]
            target_vel_k = (end_effector_pos - prev_pos) / dt
            target_acc_k = (end_effector_pos - 2 * prev_pos + prev_prev_pos) / (dt ** 2)
        elif len(trajectory) > 0:
            prev_pos = trajectory[-1][0]
            target_vel_k = (end_effector_pos - prev_pos) / dt
            target_acc_k = np.array([0, 0, 0])  # 初始加速度为零
        else:
            target_vel_k = np.array([0, 0, 0])
            target_acc_k = np.array([0, 0, 0])
        
        trajectory.append((end_effector_pos, target_vel_k, target_acc_k))
    
    return trajectory

# 可视化函数，包括机械臂和末端轨迹
def plot_trajectory_with_arm_3d(target_positions, znn_positions, arm_positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制目标轨迹
    target_x, target_y, target_z = zip(*target_positions)
    ax.plot(target_x, target_y, target_z, 'r--', label='target traj')

    # 绘制末端执行器跟踪的轨迹
    znn_x, znn_y, znn_z = zip(*znn_positions)
    ax.plot(znn_x, znn_y, znn_z, color='blue', label='ZNN end-effector traj')

    # 标记起点
    ax.scatter(target_x[0], target_y[0], target_z[0], color='red', label='start position')

    # 连续绘制机械臂的每一段，以确保两连杆机械臂的连贯性
    for base, end in arm_positions:
        ax.plot([0, base[0]], [0, base[1]], [0, base[2]], 'g-', alpha=0.2, linewidth=2)  # 第一连杆
        ax.plot([base[0], end[0]], [base[1], end[1]], [base[2], end[2]], 'y-', alpha=0.2, linewidth=2)  # 第二连杆

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D ZNN traj tracking with continuous arm movement')
    plt.show()

# 计算实际加速度
def calculate_actual_acceleration(current_pos_k, prev_pos_k, prev_prev_pos_k, dt):
    # 使用实际的位置计算加速度
    return (current_pos_k - 2 * prev_pos_k + prev_prev_pos_k) / (dt ** 2)

# 在 znn_trajectory_tracking 函数中修正部分
def znn_trajectory_tracking(initial_angles, trajectory):
    angle_k = np.copy(initial_angles)
    angle_dot_k = np.zeros(2)  # 初始化角速度
    jacobian_matrix_k = np.eye(2)  # 初始化雅克比矩阵
    jacobian_dot_k = np.zeros((2, 2))  # 初始化雅克比矩阵的变化率

    znn_positions = []
    arm_positions = []

    # 用于保存之前的位置信息以计算实际加速度
    prev_pos_k = None
    prev_prev_pos_k = None

    for target_pos_k, target_vel_k, target_acc_k in trajectory:
        base_pos_k, current_pos_k = forward_kinematics(angle_k)
        arm_positions.append((base_pos_k, current_pos_k))  # 保存机械臂位置
        
        # 计算实际加速度
        if prev_pos_k is not None and prev_prev_pos_k is not None:
            actual_acceleration_k = calculate_actual_acceleration(current_pos_k, prev_pos_k, prev_prev_pos_k, dt)
        else:
            actual_acceleration_k = np.zeros(3)  # 初始加速度为零

        # 保存位置信息供下一步计算
        prev_prev_pos_k = prev_pos_k
        prev_pos_k = current_pos_k

        # 更新角度 (通过 IENT-ZNN1)
        angle_k, angle_dot_k = znn1_update(angle_k, angle_dot_k, current_pos_k, target_pos_k, target_vel_k, jacobian_matrix_k)
        znn_positions.append(current_pos_k)

        # 更新雅克比矩阵 (通过 IENT-ZNN2)
        jacobian_matrix_k, jacobian_dot_k = znn2_update(jacobian_matrix_k, angle_dot_k, angle_dot_k, actual_acceleration_k, target_vel_k, dt=0.01)

    return angle_k, znn_positions, arm_positions


# 示例参数
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = np.pi
num_points = 500
dt = 0.01

# 生成目标弧形轨迹
trajectory = generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt)
target_positions = [pos for pos, _, _ in trajectory]

# 执行 ZNN 轨迹跟踪
angle_k, znn_positions, arm_positions = znn_trajectory_tracking(initial_angles, trajectory)

# 绘制3D对比图，包括机械臂
plot_trajectory_with_arm_3d(target_positions, znn_positions, arm_positions)
