import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义全局常量
DT = 0.001  # 全局时间步长
# BETA_ZNN1 = 20  # ZNN1 的 beta 参数
# BETA_ZNN2 = 20  # ZNN2 的 beta 参数
BETA_ZNN1 = 20  # ZNN1 的 beta 参数
BETA_ZNN2 = 20  # ZNN2 的 beta 参数

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
def znn1_update(angle_k, angle_dot_k, pos_k, target_pos_k, target_vel_k, jacobian_matrix_k, beta=BETA_ZNN1, dt=DT):
    pos_error_k = target_pos_k[:2] - pos_k[:2]
    angle_dot_k_next = safe_clip((np.eye(2) - jacobian_matrix_k.T @ jacobian_matrix_k) @ angle_dot_k + jacobian_matrix_k.T @ (target_vel_k[:2] + beta * pos_error_k))
    angle_k_next = safe_clip(angle_k + dt * angle_dot_k_next ) # 使用欧拉前向方法离散化更新角度
    return angle_k_next, angle_dot_k_next

# IENT-ZNN2 更新公式，用于估计雅克比矩

def znn2_update(jacobian_k, jacobian_dot_k,angles_dot_k, angles_dotdot_k, actual_acceleration_k, current_velocity_k, beta=BETA_ZNN2, dt=DT):
    jacobian_dot_k_next = safe_clip((
        (actual_acceleration_k[:2] - jacobian_k @ angles_dotdot_k[:2] + beta * (current_velocity_k[:2] - jacobian_k @ angles_dot_k[:2]))
        @ angles_dot_k[:2].T
        + jacobian_dot_k @ (np.eye(2) - angles_dot_k[:2] @ angles_dot_k[:2].T)
    ))
    jacobian_k_next = safe_clip(jacobian_k + dt * jacobian_dot_k_next)
    return jacobian_k_next, jacobian_dot_k_next

# 定义误差计算函数
def compute_total_error(target_positions, znn_positions):
    return np.sum(np.linalg.norm(np.array(target_positions) - np.array(znn_positions), axis=1))

# 网格搜索 beta 参数
# 修复后的 optimize_beta_parameters 函数
def optimize_beta_parameters(initial_angles, trajectory, beta1_range, beta2_range):
    best_beta1, best_beta2 = None, None
    best_error = float('inf')

    for beta_znn1 in beta1_range:
        for beta_znn2 in beta2_range:
            _, znn_positions, _ = znn_trajectory_tracking(initial_angles, trajectory, beta_znn1, beta_znn2)
            error = compute_total_error([pos for pos, _, _ in trajectory], znn_positions)

            # 添加无效值检查
            if not np.isfinite(error):
                continue

            if error < best_error:
                best_error = error
                best_beta1, best_beta2 = beta_znn1, beta_znn2

    return best_beta1, best_beta2, best_error

# 改进优化流程：分阶段逐步优化
def optimize_beta_parameters_stagewise(initial_angles, trajectory):
    # 阶段 1：粗略搜索
    beta1_range = np.linspace(10, 50, 1)
    beta2_range = np.linspace(10, 50, 1)
    best_beta1, best_beta2, best_error = optimize_beta_parameters(initial_angles, trajectory, beta1_range, beta2_range)

    print(f"Stage 1 Optimal beta_znn1: {best_beta1}, Optimal beta_znn2: {best_beta2}, Error: {best_error}")

    # 阶段 2：扩大范围
    beta1_range = np.linspace(best_beta1 - 1, best_beta1 + 1, 11)
    beta2_range = np.linspace(best_beta2 - 1, best_beta2 + 1, 11)
    best_beta1, best_beta2, best_error = optimize_beta_parameters(initial_angles, trajectory, beta1_range, beta2_range)

    print(f"Stage 2 Optimal beta_znn1: {best_beta1}, Optimal beta_znn2: {best_beta2}, Error: {best_error}")

    # 阶段 3：精细调整
    beta1_range = np.linspace(best_beta1 - 1, best_beta1 + 1, 21)
    beta2_range = np.linspace(best_beta2 - 1, best_beta2 + 1, 21)
    best_beta1, best_beta2, best_error = optimize_beta_parameters(initial_angles, trajectory, beta1_range, beta2_range)

    print(f"Stage 3 Optimal beta_znn1: {best_beta1}, Optimal beta_znn2: {best_beta2}, Error: {best_error}")

    return best_beta1, best_beta2, best_error


# 数值稳定化函数
def safe_clip(value, min_value=-1e6, max_value=1e6):
    return np.clip(value, min_value, max_value)

# 生成可达的弧形轨迹
def generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points):
    trajectory = []
    angles = np.linspace(0, delta_angle, num_points)
    
    for angle in angles:
        theta1 = initial_angles[0] + angle
        theta2 = initial_angles[1] - angle
        _, end_effector_pos = forward_kinematics([theta1, theta2])
        
        if len(trajectory) > 1:
            prev_pos = trajectory[-1][0]
            prev_prev_pos = trajectory[-2][0]
            target_vel_k = (end_effector_pos - prev_pos) / DT
            target_acc_k = (end_effector_pos - 2 * prev_pos + prev_prev_pos) / (DT ** 2)
        elif len(trajectory) > 0:
            prev_pos = trajectory[-1][0]
            target_vel_k = (end_effector_pos - prev_pos) / DT
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
def calculate_actual_acceleration(current_pos_k, prev_pos_k, prev_prev_pos_k):
    return (current_pos_k - 2 * prev_pos_k + prev_prev_pos_k) / (DT ** 2)

# 修复后的 znn_trajectory_tracking 函数
def znn_trajectory_tracking(initial_angles, trajectory, beta_znn1, beta_znn2):
    angle_k = np.copy(initial_angles)
    angle_dot_k = np.zeros(2)  # 初始化角速度
    jacobian_matrix_k = np.eye(2)  # 初始化雅克比矩阵
    jacobian_dot_k = np.zeros((2, 2))  # 初始化雅克比变化率

    znn_positions = []
    arm_positions = []

    prev_pos_k = None
    prev_prev_pos_k = None
    prev_angle_dot_k = np.zeros(2)

    for target_pos_k, target_vel_k, target_acc_k in trajectory:
        base_pos_k, current_pos_k = forward_kinematics(angle_k)
        arm_positions.append((base_pos_k, current_pos_k))  # 保存机械臂位置

        if prev_pos_k is not None and prev_prev_pos_k is not None:
            actual_acceleration_k = calculate_actual_acceleration(current_pos_k, prev_pos_k, prev_prev_pos_k)
            current_velocity_k = safe_clip((current_pos_k - prev_pos_k) / DT)
            angles_dotdot_k = safe_clip((angle_dot_k - prev_angle_dot_k) / DT)
        else:
            actual_acceleration_k = np.zeros(3)
            current_velocity_k = np.zeros(3)
            angles_dotdot_k = np.zeros(2)

        prev_prev_pos_k = prev_pos_k
        prev_pos_k = current_pos_k
        prev_angle_dot_k = angle_dot_k

        # 更新角度 (IENT-ZNN1)
        angle_k, angle_dot_k = znn1_update(angle_k, angle_dot_k, current_pos_k, target_pos_k, target_vel_k, jacobian_matrix_k, beta=beta_znn1)
        znn_positions.append(current_pos_k)

        # 更新雅克比矩阵 (IENT-ZNN2)
        jacobian_matrix_k, jacobian_dot_k = znn2_update(
            jacobian_matrix_k, jacobian_dot_k, angle_dot_k, angles_dotdot_k,
            actual_acceleration_k, current_velocity_k, beta=beta_znn2
        )

    return angle_k, znn_positions, arm_positions

# 示例参数
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = np.pi
num_points = 500

# 生成目标弧形轨迹
trajectory = generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points)
target_positions = [pos for pos, _, _ in trajectory]

# 使用改进的分阶段优化流程搜索最优 beta 参数
best_beta1, best_beta2, best_error = optimize_beta_parameters_stagewise(initial_angles, trajectory)

# 使用最优 beta 参数运行 ZNN 轨迹追踪
_, znn_positions, arm_positions = znn_trajectory_tracking(initial_angles, trajectory, best_beta1, best_beta2)

# 绘制3D对比图，包括机械臂
plot_trajectory_with_arm_3d(target_positions, znn_positions, arm_positions)

errors = [np.linalg.norm(target - znn) for target, znn in zip(target_positions, znn_positions)]
plt.plot(errors)
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Error Convergence')
plt.show()

