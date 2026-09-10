import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# ================= 导入 SDK 及环境配置 =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)


from lib.lib_h1_sdk_python import (
    H1Robot,
    MotorControlMode,
    Motor_Control,
    EtherCAT_Motor_Index,
    ArmAction,
    ArmPose,
    ArmEndPose,
    Motor_Control
)

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

def load_pose_matrix(filepath):
    """从 txt 文件加载 4x4 齐次变换矩阵"""
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row:
                matrix.append(row)
    return np.array(matrix)

# ================= 核心：带有抓取偏置的矩阵计算 =================
def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """
    计算最终的【相对抓取位姿】。
    """
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
    T4 = np.linalg.inv(T4_inv)

    # 物体在当前手臂零点坐标系下的绝对矩阵
    T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    # 抓取偏置矩阵 (Grasp Offset)
    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    # 沿着夹爪前方后退 8 厘米，预留空间
    T_grasp_local[:3, 3] = [-0.08, 0.0, 0.0] 

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat

def print_headCamera_target(robot: H1Robot):
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None) or getattr(cam_state, "PositionData", None)
    cam_quat_rel = getattr(cam_state, "rotation", None) or getattr(cam_state, "RotationData", None)
    print('cam_pos', cam_pos_rel)
    print('cam_quat', cam_quat_rel)

def print_head_target(robot: H1Robot):

    ok_head, head_state = robot.getHeadRelative()
    head_pos_rel = getattr(head_state, "position", None) or getattr(head_state, "PositionData", None)
    head_quat_rel = getattr(head_state, "rotation", None) or getattr(head_state, "RotationData", None)
    print('head_pos', head_pos_rel)
    print('head_quat', head_quat_rel)

def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[流程 A] 正在调整机器人初始观测姿态...")
    waist_steps = int(4.0 * RATE_HZ)

    for i in range(1, waist_steps + 1):
        ratio = i / waist_steps
        waist_pose = ArmPose()
        waist_pose.x = 0.0
        waist_pose.y = 0.0
        waist_pose.roll = 0.0
        waist_pose.yaw = 0.0

        # 关键修改：从当前值平滑插值到目标值
        waist_pose.z    = cur_waist_z    + (tar_waist_z    - cur_waist_z)    * ratio
        waist_pose.pitch = cur_waist_pitch + (tar_waist_pitch - cur_waist_pitch) * ratio

        end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end_pose)
        time.sleep(DT)

    time.sleep(1.5)
    print(" -> 2/2 头部云台俯视中...")
    head_pitch = Motor_Control()
    head_pitch.Position = 0.0
    target_pitch_angle = -0.1745  
    head_steps = int(1.5 * RATE_HZ) 
    
    for i in range(1, head_steps + 1):
        head_pitch.Position = target_pitch_angle * (i / head_steps)
        robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
        time.sleep(DT)
        
    time.sleep(2.0)
    print(" -> 观测姿态调整完毕。")

# ================= 主控制流程 =================
def main():
    try:
        print("[流程 1/7] 实例化机器人并尝试连接...")
        robot = H1Robot()
        if not robot.robot_connect():
            print("连接失败！")
            return

        print("[流程 2/7] 切换至 HIGH_LEVEL 模式并初始化...")
        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        time.sleep(3.0)

        # monitor = start_state_monitor(robot, EtherCAT_Motor_Index, period=1.0)
        prepare_robot_posture(robot, 0, 0, 0.6, 1.0)
        time.sleep(3.0)
        # prepare_robot_posture(robot, 0.15, 1.6, 0.15, 0.0)
        
        
        while True:
            # ok_cam, cam_state = robot.getHeadCameraRelative()
            # cam_pos_rel = getattr(cam_state, "position",None)
            # cam_quat_rel = getattr(cam_state, "rotation", None)
            # print(cam_pos_rel)
            # print(cam_quat_rel)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备执行安全下线流程...")

    finally:
        print("[任务结束] 回收机器人控制权...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()

if __name__ == "__main__":
    main()