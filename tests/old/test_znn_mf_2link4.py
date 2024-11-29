import numpy as np
import matplotlib.pyplot as plt

# 全局常量
DT = 0.001  # 时间步长
BETA_ZNN1 = 1  # ZNN1 的 beta 参数
BETA_ZNN2 = 2  # ZNN2 的 beta 参数
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
def znn1_update(angle_dot, pos, target_pos, target_vel, jacobian, beta=BETA_ZNN1):
    # 计算位置误差
    pos_error = target_pos - pos
    # 分步计算
    term1 = np.eye(2) - jacobian.T @ jacobian  # 投影调整项
    term2 = jacobian.T @ target_vel           # 目标速度项
    term3 = beta * jacobian.T @ pos_error     # 误差修正项
    # 合并结果
    angle_dot = term1 @ angle_dot + term2 + term3
    return angle_dot

# ZNN2 更新公式
def znn2_update(jacobian, jacobian_dot, angles_dot, angles_dotdot, actual_acceleration, current_velocity, beta=BETA_ZNN2):
    # 分步计算
    term1 = actual_acceleration - jacobian @ angles_dotdot  # 实际加速度项
    term2 = beta * (current_velocity - jacobian @ angles_dot)  # 误差修正项
    term3 = term1 + term2  # 合并前两项
    jacobian_dot = term3 @ angles_dot.T + jacobian_dot @ (np.eye(2) - angles_dot @ angles_dot.T)
    return jacobian_dot

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
            target_vel = (trajectory[i] - trajectory[i - 1]) / dt
            #target_vel = (trajectory[i] - trajectory[i - 1]) / 1
        else:
            target_vel = np.zeros(2)  # 初始时刻速度为0
        velocities.append(target_vel)

    return trajectory, velocities

def generate_straight_line_trajectory_via_fk(start_pos, end_pos, total_time, dt, initial_angles):
    """
    使用正向运动学生成末端沿直线路径运动的轨迹。

    参数:
    - start_pos: 直线轨迹的起点 [x_start, y_start]
    - end_pos: 直线轨迹的终点 [x_end, y_end]
    - total_time: 总运动时间
    - dt: 时间步长
    - initial_angles: 初始关节角度 [theta1, theta2]

    返回:
    - trajectory: 末端执行器的轨迹点列表
    - velocities: 末端执行器的目标速度列表
    """
    num_points = int(total_time / dt)
    trajectory = []
    velocities = []
    
    # 在直线段中生成点
    x_values = np.linspace(start_pos[0], end_pos[0], num_points)
    y_values = np.linspace(start_pos[1], end_pos[1], num_points)
    
    current_angles = np.copy(initial_angles)
    
    for i in range(num_points):
        # 目标位置
        target_pos = np.array([x_values[i], y_values[i]])
        
        # 使用逆运动学找到关节角度（假设二维平面两连杆机器人）
        l1, l2 = 1.0, 1.0  # 连杆长度
        x, y = target_pos
        
        # 计算关节角度（逆运动学）
        cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(cos_theta2) > 1.0:
            raise ValueError("目标位置超出机械臂的可达范围")
        sin_theta2 = np.sqrt(1 - cos_theta2**2)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        
        k1 = l1 + l2 * cos_theta2
        k2 = l2 * sin_theta2
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        current_angles = [theta1, theta2]
        
        # 使用正向运动学计算当前末端位置
        _, end_effector_pos = forward_kinematics(current_angles)
        trajectory.append(end_effector_pos)
        
        # 计算目标速度
        if i > 0:
            target_vel = (trajectory[i] - trajectory[i - 1]) / dt
        else:
            target_vel = np.zeros(2)
        velocities.append(target_vel)
    
    return trajectory, velocities


def znn_trajectory_tracking(initial_angles, trajectory, velocities, dt):
    angles = np.copy(initial_angles)
    angle_dot = np.zeros(2)  # 初始关节速度
    jacobian = initialize_jacobian(angles)  # 初始雅克比矩阵
    jacobian_dot = np.zeros_like(jacobian)  # 初始雅克比变化率

    znn_positions = []

    prev_pos = None
    prev_vel = np.zeros(2)  # 初始末端速度
    prev_angle_dot = np.zeros(2)  # 初始关节速度

    for k in range(len(trajectory)):  # 使用轨迹长度直接控制循环
        # 当前目标位置和速度
        target_pos = trajectory[k]
        target_vel = velocities[k]

        # 计算当前末端位置
        _, current_pos = forward_kinematics(angles)
        znn_positions.append(current_pos)

        # ZNN1 计算关节角速度
        angle_dot = znn1_update(angle_dot, current_pos, target_pos, target_vel, jacobian)
        #angle_dot = np.clip(angle_dot, -100, 100)  # 限制角速度范围
        # 离散更新关节角度
        angles = discrete_update(angles, angle_dot, dt)
         # **对角度进行限制，防止累积误差**
        #angles = np.mod(angles, 2 * np.pi)  # 限制角度在 [-2π, 2π]

        # 计算末端速度
        if prev_pos is not None:
            actual_vel = (current_pos - prev_pos) / dt
        else:
            actual_vel = np.zeros(2)

        # 计算末端加速度
        if prev_pos is not None:
            actual_acc = calculate_acceleration(actual_vel, prev_vel, dt)
        else:
            actual_acc = np.zeros(2)

        # 计算关节加速度
        angles_dotdot = calculate_acceleration(angle_dot, prev_angle_dot, dt)

        # ZNN2 更新雅克比变化率
        # jacobian_dot = znn2_update(jacobian, jacobian_dot, angle_dot, angles_dotdot, actual_acc, actual_vel)

        # # 离散更新雅克比矩阵
        # jacobian = discrete_update(jacobian, jacobian_dot, dt)
        jacobian = initialize_jacobian(angles)

        # 保存上一时刻的值
        prev_pos = current_pos
        prev_vel = actual_vel
        prev_angle_dot = angle_dot

    return znn_positions


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
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = np.pi / 2
start_pos = [0, 2]  # 起点
end_pos = [2, 0]    # 终点



#trajectory, velocities = generate_trajectory_with_velocity(initial_angles, delta_angle, TD, DT)
trajectory, velocities = generate_straight_line_trajectory_via_fk(start_pos, end_pos, TD, DT, initial_angles)

znn_positions = znn_trajectory_tracking(initial_angles, trajectory, velocities, DT)
plot_trajectory(trajectory, znn_positions)
trajectory_array = np.array(trajectory)
plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'r--', label='Target Straight Line Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Generated Straight Line Trajectory')
plt.axis('equal')
plt.show()

