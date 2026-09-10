import os, sys, time

# 将 lib 目录加入 sys.path 以便导入编译出的 lib_h1_sdk_python.so
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 不要加入 lib 目录，而是加入 root 目录
if root not in sys.path:
    sys.path.insert(0, root)



from lib.lib_h1_sdk_python import (
    H1Robot,
    MotorControlMode,
    Motor_Control,
    EtherCAT_Motor_Index,
    ArmAction,       # 枚举: LEFT_ARM / RIGHT_ARM
    ArmPose,         # 三维数据+欧拉姿态输入
    ArmEndPose       # 三维数据+四元数姿态输入
)

RATE_HZ = 500
DT = 1.0 / RATE_HZ

def chassis_move(robot: H1Robot):
    """
    以0.2m/s的速度前进1米
    """
    DT = 0.2  # 200ms发送一次指令
    SPEED = 0.2  # 目标速度 m/s
    DISTANCE = 0.6  # 目标距离 米
    DURATION = DISTANCE / SPEED  # 持续时间 = 5秒
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            
            # 发送底盘速度指令
            robot.setChassis_high(-SPEED, 0.0)  # speed_x=0.2, speed_yaw=0.0
            
            # 计算已经过时间
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            # 维持0.2s间隔
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 到达目标位置，停止
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进1米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)  # 停止

def chassis_rotation(robot: H1Robot):
    """
    以0.2m/s的速度前进1米
    """
    DT = 0.2  # 200ms发送一次指令
    SPEED = 0.2  # 目标速度 m/s
    DISTANCE = 1.0  # 目标距离 米
    DURATION = DISTANCE / SPEED  # 持续时间 = 5秒
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            
            # 发送底盘速度指令
            robot.setChassis_high(SPEED, 0.0)  # speed_x=0.2, speed_yaw=0.0
            
            # 计算已经过时间
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            # 维持0.2s间隔
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 到达目标位置，停止
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进1米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)  # 停止

def rotate_90_degrees(robot: H1Robot, direction="left"):
    """
    旋转90度
    底盘角速度单位是 rad/s
    
    Args:
        direction: "left" 或 "right"
    """
    DT = 0.2
    ANGULAR_SPEED = 0.5  # 角速度 rad/s (可根据需要调整)
    TARGET_ANGLE = 1.5708  # 90度 = π/2 弧度
    DURATION = TARGET_ANGLE / ANGULAR_SPEED + 1.2  # ≈ 3.14秒
    
    # 确定旋转方向
    if direction == "left":
        yaw_speed = ANGULAR_SPEED  # 正值为左转
    else:
        yaw_speed = -ANGULAR_SPEED  # 负值为右转
    
    print(f"[Chassis] 旋转90度{direction}, 角速度{ANGULAR_SPEED}rad/s, 持续{DURATION:.2f}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            
            # 设置角速度，线速度为0
            robot.setChassis_high(0.0, yaw_speed)
            
            elapsed = time.time() - start_time
            rotated_angle = ANGULAR_SPEED * elapsed
            rotated_degrees = rotated_angle * 180 / 3.14159
            print(f"[Chassis] 已旋转: {rotated_degrees:.1f}°")
            
            # 维持0.2s间隔
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time = DT - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 旋转90度完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 中断")
        robot.setChassis_high(0.0, 0.0)

def main():
    try:
        robot = H1Robot()

        if not robot.robot_connect():
            print("connect failed")
            return
        print("connected =", robot.isRobotConnected())

        if not robot.switchControlMode(MotorControlMode.HIGH_LEVEL):
            print("switch HIGH_LEVEL failed")
            return
        print("mode =", robot.getCurrentMode())

        if not robot.robot_init():
            print("robot_init failed")
            return
        print("robot initialized")
        time.sleep(1.0)

        chassis_move(robot)
        # time.sleep(2.0)
        # rotate_90_degrees(robot, "right")
        # time.sleep(2.0)
        # chassis_move(robot)
        while True:
            time.sleep(1.0)


    
    except KeyboardInterrupt:
        print("捕获到ctrl+C 中断 (未做清理)")
    finally:
        # return
        # 若要恢复优雅退出, 删除上面 return 并取消注释:
        if hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
            print("done")

if __name__ == "__main__":
    main()