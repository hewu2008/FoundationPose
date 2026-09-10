import os
import sys
import time
import cv2
import zmq
import pickle
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# ================= 导入 SDK 及环境配置 =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

# 机器人 SDK
from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode, EtherCAT_Motor_Index,
    ArmAction, ArmPose, ArmEndPose, Motor_Control
)

# 相机 gRPC 客户端 (需确保与本脚本路径层级兼容)
from camera_client import CameraClient

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

# FoundationPose & Zerith 配置
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
ZMQ_SERVER_ADDR = "tcp://172.31.200.250:5555"
MESH_FILE = "assets/DPPUB-204001196-AAX_01_01.obj"
TARGET_LABELS = ["translucent white brake fluid reservoir"]



# ================= 2. 机器人姿态准备模块 =================
def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[流程 A] 正在调整机器人初始观测姿态...")
    waist_steps = int(4.0 * RATE_HZ)

    waist_pose = ArmPose()
    waist_pose.x = waist_pose.y = 0.0
    waist_pose.z = cur_waist_z
    waist_pose.roll = 0.0
    waist_pose.pitch = cur_waist_pitch
    waist_pose.yaw = 0.0

    diff_z = tar_waist_z - cur_waist_z
    diff_pitch = tar_waist_pitch - cur_waist_pitch

    def send():
        end = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end)

    for _ in range(1, waist_steps + 1):
        waist_pose.z += diff_z / waist_steps
        send(); time.sleep(DT)    
    
    for _ in range(1, waist_steps + 1):
        waist_pose.pitch += diff_pitch / waist_steps
        send(); time.sleep(DT)

    time.sleep(1.5)

    # # 动作 2：头部俯视
    # print(" -> 2/2 头部云台俯视中...")
    # head_pitch = Motor_Control()
    # head_pitch.Position = 0.0
    # target_pitch_angle = 0.3  
    # head_steps = int(1.5 * RATE_HZ) 
    
    # for i in range(1, head_steps + 1):
    #     head_pitch.Position = target_pitch_angle * (i / head_steps)
    #     robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
    #     time.sleep(DT)
        
    # time.sleep(2.0)
    print(" -> 观测姿态调整完毕。")

def armright_move(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    duration_arm = 4.0
    steps_arm = int(duration_arm * RATE_HZ)

    start_x, start_y, start_z = cur_xyz[0], cur_xyz[1], cur_xyz[2]
    start_quat = cur_quat

    dest_x, dest_y, dest_z = dest_xyz[0], dest_xyz[1], dest_xyz[2] 
    # dest_quat = [0.0, 0.0, 0.0, 1.0]
    # dest_quat = [0.0000, -0.2588, 0.0000, 0.9659]

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps_arm + 1):
        ratio = i / steps_arm

        x = start_x + (dest_x - start_x) * ratio
        y = start_y + (dest_y - start_y) * ratio
        z = start_z + (dest_z - start_z) * ratio

        interp_quat = slerp(ratio).as_quat()

        target_end_pose = ArmEndPose()
        target_end_pose.position = [x, y, z]
        target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        robot.setArm_high(ArmAction.RIGHT_ARM, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def armleft_move(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    duration_arm = 4.0
    steps_arm = int(duration_arm * RATE_HZ)

    start_x, start_y, start_z = cur_xyz[0], cur_xyz[1], cur_xyz[2]
    start_quat = cur_quat

    dest_x, dest_y, dest_z = dest_xyz[0], dest_xyz[1], dest_xyz[2] 
    # dest_quat = [0.0, 0.0, 0.0, 1.0]
    dest_quat = [0.0000, -0.2588, 0.0000, 0.9659]

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps_arm + 1):
        ratio = i / steps_arm

        x = start_x + (dest_x - start_x) * ratio
        y = start_y + (dest_y - start_y) * ratio
        z = start_z + (dest_z - start_z) * ratio

        interp_quat = slerp(ratio).as_quat()

        target_end_pose = ArmEndPose()
        target_end_pose.position = [x, y, z]
        target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def chassis_back(robot: H1Robot):
    DT = 0.2  
    SPEED = 0.2  
    DISTANCE = -0.3  
    DURATION = DISTANCE / SPEED  
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            robot.setChassis_high(SPEED, 0.0)  
            
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进1米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)

def chassis_move(robot: H1Robot):
    DT = 0.2  
    SPEED = 0.2  
    DISTANCE = 0.3  
    DURATION = DISTANCE / SPEED  
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            robot.setChassis_high(SPEED, 0.0)  
            
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进1米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)

# ================= 5. 抓取运算与执行模块 =================
def load_pose_matrix(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row: matrix.append(row)
    return np.array(matrix)

def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """坐标系变换：感知位姿 -> 手臂相对零点目标位姿"""
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T_comp = np.eye(4)
    
    #T_comp[:3, 3] = [-0.03, -0.05, -0.02]
    #T_comp[:3, 3] = [0.12, -0.01, 0.12]
    T4_inv = np.dot(T4_inv, T_comp)
    T4 = np.linalg.inv(T4_inv)

    T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] 

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat



# ================= 主控制流 =================
def main():
    robot = H1Robot()
    try:
        print("[INIT] 实例化机器人并连接...")
        if not robot.robot_connect():
            print("连接机器人失败！")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        time.sleep(3.0)

        # ------------------------------------------------
        # 1. 机器人姿态准备 (升高、弯腰、低头)
        # ------------------------------------------------
        #chassis_back(robot)
        time.sleep(1.5)
        prepare_robot_posture(robot, 0, 0, 0.6, 1.0)


        armleft_move(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.02, 0.30],
                [0.0, 0.0, 0.0, 1])
        armright_move(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, -0.02, 0.30],
                [0.0, 0.0, 0.0, 1])
        time.sleep(2.0)

        #chassis_move(robot)
        time.sleep(2.0)
        # ------------------------------------------------
        # 4. 读取手臂/相机的当前状态并解算最终目标点
        # ------------------------------------------------
        print("\n[流程 D] 读取位姿状态并进行矩阵解算...")
        ok_cam, cam_state = robot.getHeadCameraRelative()
        cam_pos_rel = getattr(cam_state, "position", None) 
        cam_quat_rel = getattr(cam_state, "rotation", None) 

        ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        #ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
        arm_pos_rel = getattr(arm_state, "position", None) 
        arm_quat_rel = getattr(arm_state, "rotation", None) 

        if not (ok_cam and ok_arm):
            print("传感器位姿获取失败！")
        

        T_obj_cam = load_pose_matrix("0.txt")
        target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam)
        print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

        
        # ------------------------------------------------
        # 5. 执行手臂平移逼近目标
        # ------------------------------------------------
        
        # [0.0000, -0.2588, 0.0000, 0.9659]
        # X=0.2522, Y=-0.1120, Z=-0.0698
        # X=0.1684, Y=-0.1141, Z=-0.0075
        #prepare_robot_posture(robot, 0.6, 1.0, 0.35, 1.0)
        time.sleep(1.5)
        # ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        # arm_pos_rel = getattr(arm_state, "position", None)
        # target_pos = [arm_pos_rel[0]+0.1684, arm_pos_rel[1]-0.1141, arm_pos_rel[2]-0.0075]
        armleft_move(robot, arm_pos_rel, [0.0, 0.0, 0.0, 1.0], target_pos,
            [0.0, 0.0, 0.0, 1])

        # ------------------------------------------------
        # 6. 闭合夹爪进行抓取
        # ------------------------------------------------
        print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
        close_cmd = Motor_Control()
        close_cmd.Position = 1.5
        robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)


        time.sleep(2.0)
        print(" -> 已平滑到达目标点！")

        # open_cmd = Motor_Control()
        # open_cmd.Position = 0.0
        # robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, open_cmd)


        # ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        # arm_pos_rel = getattr(arm_state, "position", None)
        # arm_quat_rel = getattr(arm_state, "rotation", None)

        # armleft_move(robot, arm_pos_rel, arm_quat_rel, [0, 0, 0], [0, 0, 0, 1])


        while True:
            # ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
            # arm_pos_rel = getattr(arm_state, "position", None)
            # arm_quat_rel = getattr(arm_state, "rotation", None)
            # print(arm_pos_rel)
            # print(arm_quat_rel)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备执行安全下线流程...")
    except Exception as e:
        print(f"\n[!] 运行时发生异常: {e}")

    finally:
        print("[清理] 回收机器人控制权...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()

if __name__ == "__main__":
    main()