import numpy as np
import matplotlib.pyplot as plt

# 定义两连杆的正向运动学
def forward_kinematics(angles):
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = angles
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])

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
    pos_error = actual_pos - desired_pos
    
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
        end_effector_pos = forward_kinematics([theta1, theta2])
        
        if len(trajectory) > 0:
            prev_pos, _ = trajectory[-1]
            target_vel = (end_effector_pos - prev_pos) / dt
        else:
            target_vel = np.array([0, 0])
        
        trajectory.append((end_effector_pos, target_vel))
    
    return trajectory

# 可视化函数
def plot_trajectory(target_positions, znn_positions):
    target_x, target_y = zip(*target_positions)
    znn_x, znn_y = zip(*znn_positions)

    plt.figure(figsize=(8, 8))
    plt.plot(target_x, target_y, 'r--', label='target traj')
    plt.plot(znn_x, znn_y, 'b-', label='ZNN')
    plt.scatter(target_x[0], target_y[0], color='red', label='start position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('ZNN traj tracking')
    plt.axis('equal')
    plt.show()

# 执行 ZNN 轨迹跟踪
def znn_trajectory_tracking(initial_angles, trajectory):
    angles = np.copy(initial_angles)
    znn_positions = []

    for target_pos, target_vel in trajectory:
        current_pos = forward_kinematics(angles)

        angles = znn_update(angles, current_pos, target_pos, target_vel, tau=0.0001, h=1) # 0.01 to 1.5 good
        #angles = znn_update(angles, current_pos, target_pos, target_vel, tau=0.0001, h=0.0001)
        znn_positions.append(current_pos)

    return angles, znn_positions

# 示例参数
initial_angles = np.array([np.pi / 4, np.pi / 4])
delta_angle = 2*np.pi
num_points = 500
dt = 0.0001

# 生成目标弧形轨迹
trajectory = generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt)
target_positions = [pos for pos, _ in trajectory]

# 执行 ZNN 轨迹跟踪
angles, znn_positions = znn_trajectory_tracking(initial_angles, trajectory)

# 绘制对比图
plot_trajectory(target_positions, znn_positions)


# import numpy as np
# import matplotlib.pyplot as plt

# # 定义两连杆的正向运动学
# def forward_kinematics(angles):
#     l1, l2 = 1.0, 1.0  # 连杆长度
#     theta1, theta2 = angles
#     x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
#     y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
#     return np.array([x, y])

# # 计算雅克比矩阵
# def jacobian(angles):
#     l1, l2 = 1.0, 1.0
#     theta1, theta2 = angles
#     j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
#     j12 = -l2 * np.sin(theta1 + theta2)
#     j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
#     j22 = l2 * np.cos(theta1 + theta2)
#     return np.array([[j11, j12], [j21, j22]])

# # # ZNN 更新公式，基于公式(14)
# # def znn_update(angles, pos_error, desired_velocity, h=0.01, lambda_gain=1.0):
# #     J = jacobian(angles)
# #     J_pseudo = np.linalg.pinv(J)  # 计算伪逆
# #     angles_dot = J_pseudo @ (desired_velocity - lambda_gain * pos_error)
# #     return angles + h * angles_dot

# # ZNN 更新公式，严格按照公式 (14) 实现
# def znn_update(angles, actual_pos, desired_pos, desired_velocity, tau=1.0, h=1):
#     # 计算雅克比矩阵及其伪逆
#     J = jacobian(angles)
#     J_pseudo = np.linalg.pinv(J)
    
#     # 计算实际位置与期望位置的误差
#     pos_error = actual_pos - desired_pos
    
#     # 计算更新项，按照公式 (14)
#     correction_term = J_pseudo @ (tau * desired_velocity - h * pos_error)
    
#     # 更新角度，直接加上校正项
#     next_angles = angles + correction_term
    
#     return next_angles

# # 生成可达的弧形轨迹
# def generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt):
#     trajectory = []
#     angles = np.linspace(0, delta_angle, num_points)
    
#     for angle in angles:
#         theta1 = initial_angles[0] + angle
#         theta2 = initial_angles[1] - angle
#         end_effector_pos = forward_kinematics([theta1, theta2])
        
#         if len(trajectory) > 0:
#             prev_pos, _ = trajectory[-1]
#             target_vel = (end_effector_pos - prev_pos) / dt
#         else:
#             target_vel = np.array([0, 0])
        
#         trajectory.append((end_effector_pos, target_vel))
    
#     return trajectory

# # 可视化函数
# def plot_trajectory(target_positions, znn_positions):
#     target_x, target_y = zip(*target_positions)
#     znn_x, znn_y = zip(*znn_positions)

#     plt.figure(figsize=(8, 8))
#     plt.plot(target_x, target_y, 'r--', label='target traj')
#     plt.plot(znn_x, znn_y, 'b-', label='ZNN')
#     plt.scatter(target_x[0], target_y[0], color='red', label='start position')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#     plt.title('ZNN traj tracking')
#     plt.axis('equal')
#     plt.show()

# # 执行 ZNN 轨迹跟踪
# def znn_trajectory_tracking(initial_angles, trajectory, dt, lambda_gain=1.0):
#     angles = np.copy(initial_angles)
#     znn_positions = []

#     for target_pos, target_vel in trajectory:
#         current_pos = forward_kinematics(angles)
#         #pos_error = target_pos - current_pos
        
#         # 使用 ZNN 更新公式预测下一时刻的关节角度
#        # angles = znn_update(angles, pos_error, target_vel, dt, lambda_gain)
#         angles = znn_update(angles, current_pos, target_pos, target_vel, tau=1, h=1)
#         znn_positions.append(current_pos)

#     return angles, znn_positions

# # 示例参数
# initial_angles = np.array([np.pi / 4, np.pi / 4])
# #delta_angle = np.pi / 3
# delta_angle = 2*np.pi
# num_points = 100
# dt = 0.001

# # 生成目标弧形轨迹
# trajectory = generate_reachable_arc_trajectory(initial_angles, delta_angle, num_points, dt)
# target_positions = [pos for pos, _ in trajectory]

# # 执行 ZNN 轨迹跟踪
# angles, znn_positions = znn_trajectory_tracking(initial_angles, trajectory, dt)

# # 绘制对比图
# plot_trajectory(target_positions, znn_positions)