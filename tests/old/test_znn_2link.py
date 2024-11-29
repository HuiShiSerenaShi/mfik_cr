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

# 计算雅克比矩阵
def jacobian(angles):
    l1, l2 = 1.0, 1.0
    theta1, theta2 = angles
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)
    return np.array([[j11, j12], [j21, j22]])

# ZNN 更新公式，严格按照公式 (14) 实现
def znn_update(angles, actual_pos, desired_pos, desired_velocity, tau, h):
    # 计算雅克比矩阵及其伪逆
    J = jacobian(angles)
    J_pseudo = np.linalg.pinv(J)
    
    # 计算实际位置与期望位置的误差
    pos_error = actual_pos[:2] - desired_pos[:2]
    desired_velocity = desired_velocity[:2]
    
    # 计算更新项，按照公式 (14)
    correction_term = J_pseudo @ (tau * desired_velocity - h * pos_error)
    
    # 更新角度，直接加上校正项
    next_angles = angles + correction_term
    
    return next_angles

# 生成可达的弧形轨迹
def generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt):
    trajectory = []
    angles = np.linspace(0, delta_angle, num_points)
    
    for angle in angles:
        theta1 = initial_angles[0] + angle
        theta2 = initial_angles[1] - angle
        _, end_effector_pos = forward_kinematics([theta1, theta2])
        
        if len(trajectory) > 0:
            prev_pos, _ = trajectory[-1]
            target_vel = (end_effector_pos - prev_pos) / dt
        else:
            target_vel = np.array([0, 0, 0])
        
        trajectory.append((end_effector_pos, target_vel))
    
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
    ax.plot(znn_x, znn_y, znn_z, 'y-', label='ZNN end-effector traj')

    # 标记起点
    ax.scatter(target_x[0], target_y[0], target_z[0], color='red', label='start position')

    # 连续绘制机械臂的每一段，以确保两连杆机械臂的连贯性
    for base, end in arm_positions:
        # 第一段用绿色，第二段用蓝色，确保看起来是连贯的
        ax.plot([0, base[0]], [0, base[1]], [0, base[2]], 'g-', alpha=0.2, linewidth=2)  # 第一连杆
        ax.plot([base[0], end[0]], [base[1], end[1]], [base[2], end[2]], 'b-', alpha=0.2, linewidth=2)  # 第二连杆

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D ZNN traj tracking with continuous arm movement')
    plt.show()

# 执行 ZNN 轨迹跟踪
def znn_trajectory_tracking(initial_angles, trajectory):
    angles = np.copy(initial_angles)
    znn_positions = []
    arm_positions = []

    for target_pos, target_vel in trajectory:
        base_pos, current_pos = forward_kinematics(angles)
        arm_positions.append((base_pos, current_pos))  # 保存机械臂位置
        angles = znn_update(angles, current_pos, target_pos, target_vel, tau=0.0001, h=1)
        #angles = znn_update(angles, current_pos, target_pos, target_vel, tau=0.0001, h=0.0001)
        znn_positions.append(current_pos)

    return angles, znn_positions, arm_positions

# 示例参数
initial_angles = np.array([np.pi / 4, np.pi / 4])
#delta_angle = 2 * np.pi
delta_angle = np.pi
num_points = 500
dt = 0.0001

# 生成目标弧形轨迹
trajectory = generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt)
target_positions = [pos for pos, _ in trajectory]

# 执行 ZNN 轨迹跟踪
angles, znn_positions, arm_positions = znn_trajectory_tracking(initial_angles, trajectory)

# 绘制3D对比图，包括机械臂
plot_trajectory_with_arm_3d(target_positions, znn_positions, arm_positions)
