import sys
sys.path.append('/home/serenashi/thesis/tdcr-modeling/c++/build')

# import continuum_robot

# robot = continuum_robot.createRobot()
# print(robot)
# ee_frame = continuum_robot.runKinematics(robot)
# print("End effector frame:", ee_frame)

# import test_pybind11
# import time
# success = test_pybind11.test_pybind11()
# print(success)

import test_pybind11
import time

for i in range(10):
    start_time = time.time()  # 记录开始时间
    success = test_pybind11.test_pybind11()  # 调用测试函数
    end_time = time.time()  # 记录结束时间
    print(f"Iteration {i+1}: Success={success}, Time Taken={end_time - start_time} seconds")
