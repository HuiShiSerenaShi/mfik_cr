import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 全局常量
DT = 0.001  # 时间步长
BETA_ZNN1 = 1  # 初始 ZNN1 的 beta 参数
TD = 20  # 总时间（控制时长）

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
def initialize_jacobian(angles):
    l1, l2 = 1.0, 1.0
    theta1, theta2 = angles
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)
    return np.array([[j11, j12], [j21, j22]])

def znn1_update(angle_dot, pos, target_pos, target_vel, jacobian, beta):
    # 计算位置误差
    pos_error = target_pos - pos
    # 分步计算
    term1 = np.eye(2) - jacobian.T @ jacobian  # 投影调整项
    term2 = jacobian.T @ target_vel           # 目标速度项
    term3 = beta * jacobian.T @ pos_error     # 误差修正项
    # 合并结果
    angle_dot = term1 @ angle_dot + term2 + term3
    return angle_dot

# 离散更新公式
def discrete_update(current, derivative, dt=DT):
    return current + dt * derivative

# 生成直线轨迹和目标速度
def generate_straight_line_trajectory_via_fk(start_pos, end_pos, total_time, dt, initial_angles):
    num_points = int(total_time / dt)
    trajectory = []
    velocities = []
    
    # 在直线段中生成点
    x_values = np.linspace(start_pos[0], end_pos[0], num_points)
    y_values = np.linspace(start_pos[1], end_pos[1], num_points)
    
    for i in range(num_points):
        target_pos = np.array([x_values[i], y_values[i]])
        trajectory.append(target_pos)
        # 计算目标速度
        if i > 0:
            target_vel = (trajectory[i] - trajectory[i - 1]) / dt
        else:
            target_vel = np.zeros(2)
        velocities.append(target_vel)
    
    return trajectory, velocities

# ZNN 轨迹追踪
def znn_trajectory_tracking(initial_angles, trajectory, velocities, beta, dt):
    angles = np.copy(initial_angles)
    angle_dot = np.zeros(2)  # 初始关节速度
    jacobian = initialize_jacobian(angles)  # 初始雅克比矩阵

    znn_positions = []
    angles_history = []
     # 确保初始末端位置与目标轨迹起点一致
    _, initial_end_effector_pos = forward_kinematics(angles)
    if not np.allclose(initial_end_effector_pos, trajectory[0], atol=1e-3):
        print("Warning: Initial end-effector position does not match the target trajectory start.")
        print(f"Initial end-effector position: {initial_end_effector_pos}, Target start: {trajectory[0]}")


    for k in range(len(trajectory)):
        target_pos = trajectory[k]
        target_vel = velocities[k]

        _, current_pos = forward_kinematics(angles)
        znn_positions.append(current_pos)
        angles_history.append(angles)

                # 打印调试信息
        if k % 50 == 0:  # 每隔 50 步打印一次
            pos_error = target_pos - current_pos
            print(f"Step {k}:")
            print(f"  Current Pos: {current_pos}, Target Pos: {target_pos}, Error: {pos_error}")
            print(f"  Angles: {angles}")
            print(f"  Jacobian: \n{jacobian}")

        # ZNN1 更新关节角速度
        angle_dot = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian, beta)
        angle_dot = np.clip(angle_dot, -100, 100)  # 限制角速度范围
        angles = discrete_update(angles, angle_dot, dt)
        angles = np.mod(angles, 2 * np.pi)  # 将角度限制在 [0, 2π]
        jacobian = initialize_jacobian(angles)

    return znn_positions, angles_history

# 误差计算
def calculate_tracking_error(target_trajectory, znn_trajectory):
    return np.mean([np.linalg.norm(t - z) for t, z in zip(target_trajectory, znn_trajectory)])

# 动态可视化机械臂运动
def visualize_arm(trajectory, znn_positions, angles_history):
    fig, ax = plt.subplots()
    x_vals = [pos[0] for pos in trajectory]
    y_vals = [pos[1] for pos in trajectory]
    ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
    ax.set_ylim(min(y_vals) - 0.5, max(y_vals) + 0.5)
    ax.set_aspect('equal')
    target_x, target_y = zip(*trajectory)
    ax.plot(target_x, target_y, 'g--', label='Target Trajectory')
    ax.scatter(target_x[0], target_y[0], color='orange', label='Start Point', zorder=5)
    line1, = ax.plot([], [], 'ro-', lw=2, label='Link 1')
    line2, = ax.plot([], [], 'bo-', lw=2, label='Link 2')
    tracked_line, = ax.plot([], [], 'b-', lw=1, label='ZNN Trajectory')

    def update(frame):
        if frame > 0:
            tracked_line.set_data(*zip(*znn_positions[:frame]))
        _, joint2_pos = forward_kinematics(angles_history[frame])
        base_pos = np.array([0, 0])
        joint1_pos = np.array(forward_kinematics([angles_history[frame][0], 0])[0])
        line1.set_data([base_pos[0], joint1_pos[0]], [base_pos[1], joint1_pos[1]])
        line2.set_data([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]])
        return line1, line2, tracked_line

    anim = FuncAnimation(fig, update, frames=len(znn_positions), blit=True, interval=50, repeat=False)
    plt.legend()
    plt.show()


# 主程序：搜索 beta1
beta1_values = np.arange(0.5, 3.0, 0.1)  # beta1 从 0.5 到 3.0，步长 0.1
initial_angles = np.array([np.pi / 4, np.pi / 4])
start_pos = [0, 2]
end_pos = [2, 0]

trajectory, velocities = generate_straight_line_trajectory_via_fk(start_pos, end_pos, TD, DT, initial_angles)

best_beta1 = None
min_error = float('inf')
errors = []

errors = []
for beta1 in beta1_values:
    print(f"Testing Beta1={beta1:.2f}")
    znn_positions, angles_history = znn_trajectory_tracking(initial_angles, trajectory, velocities, beta1, DT)
    visualize_arm(trajectory, znn_positions, angles_history)
    error = calculate_tracking_error(trajectory, znn_positions)
    errors.append((beta1, error))
    print(f"Beta1={beta1:.2f}, Error={error:.4f}")

beta1_list, error_list = zip(*errors)
plt.plot(beta1_list, error_list, 'o-', label='Tracking Error')
plt.xlabel('Beta1')
plt.ylabel('Tracking Error')
plt.title('Error vs Beta1')
plt.legend()
plt.show()