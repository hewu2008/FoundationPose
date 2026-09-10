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
    T_comp = np.eye(4)
    T_comp[:3, 3] = [0.0, 0.0, 0.0]
    T4_inv = np.dot(T4_inv, T_comp)
    T4 = np.linalg.inv(T4_inv)


    #  13   7    5
    #  10   4    8  
    # 物体在当前手臂零点坐标系下的绝对矩阵
    T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    # 抓取偏置矩阵 (Grasp Offset)
    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    # 沿着夹爪前方后退 8 厘米，预留空间
    T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] #[-0.16, -0.013, 0.0]

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat

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

        # ----------------------------------------------------
        # 动作一：控制腰部同时进行【上升 20cm】与【前倾俯视】
        # ----------------------------------------------------
        print("[流程 3/7] 控制躯干平滑上升并前倾...")
        waist_pose = ArmPose()
        waist_pose.x = waist_pose.y = waist_pose.z = 0.0
        waist_pose.roll = waist_pose.pitch = waist_pose.yaw = 0.0
        
        target_waist_z = 0.15     # 上升 0.2 米 (20cm)
        target_waist_pitch = 1.0 # 俯仰角约 17 度 (注: 若机器人实际向后倒，请改为 -0.3)
        
        waist_duration = 2.0     # 耗时 2 秒
        waist_steps = int(waist_duration * RATE_HZ) 
        
        for i in range(1, waist_steps + 1):
            ratio = i / waist_steps
            # 躯干 Z 轴与 Pitch 轴同步平滑插值
            waist_pose.z = target_waist_z * ratio
            waist_pose.pitch = target_waist_pitch * ratio
            
            end_pose = robot.armPoseToArmEndPose(waist_pose)
            robot.setWaist_high(end_pose)
            time.sleep(DT)
            
        time.sleep(1.5) # 给定充足的时间让躯干彻底停稳止晃

        # ----------------------------------------------------
        # 动作二：控制头部俯视 (Pitch)
        # ----------------------------------------------------
        print("[流程 4/5] 控制头部平滑向下俯视...")
        head_pitch = Motor_Control()
        head_pitch.Position = 0.0
        
        target_pitch_angle = 0.1  # 0.3 弧度，约 17 度。可根据实际相机的视野范围调整大小
        head_steps = int(1.5 * RATE_HZ) # 耗时 1.5 秒
        
        for i in range(1, head_steps + 1):
            head_pitch.Position = target_pitch_angle * (i / head_steps)
            robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
            time.sleep(DT)
            
        time.sleep(2.0) # 等待头部云台稳定

        # ----------------------------------------------------
        # 动作三：获取【躯干姿态改变后】的传感器最新数据
        # ----------------------------------------------------
        print("[流程 4/7] 获取最新相机与手臂位姿...")
        ok_cam, cam_state = robot.getHeadCameraRelative()
        cam_pos_rel = getattr(cam_state, "position", None) or getattr(cam_state, "PositionData", None)
        cam_quat_rel = getattr(cam_state, "rotation", None) or getattr(cam_state, "RotationData", None)

        ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        arm_pos_rel = getattr(arm_state, "position", None) or getattr(arm_state, "PositionData", None)
        arm_quat_rel = getattr(arm_state, "rotation", None) or getattr(arm_state, "RotationData", None)

        if not (ok_cam and ok_arm):
            print("传感器位姿获取失败！")
            return

        # ----------------------------------------------------
        # 动作四：读取视觉姿态并进行矩阵解算
        # ----------------------------------------------------
        print("[流程 5/7] 正在解算相对抓取位姿...")
        pose_file = "0.txt"
        if not os.path.exists(pose_file):
            print(f"[错误] 找不到视觉位姿文件: {pose_file}")
            return

        T_obj_cam = load_pose_matrix(pose_file)

        target_pos, target_quat = calculate_target_relative_pose(
            cam_pos_rel, cam_quat_rel,
            arm_pos_rel, arm_quat_rel,
            T_obj_cam
        )
        print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

        # ----------------------------------------------------
        # 动作五：手臂平滑移动（严格遵照相对零点规则）
        # ----------------------------------------------------
        print("[流程 6/7] 控制手臂从相对零点平滑逼近目标...")
        
        duration_arm = 4.0
        steps_arm = int(duration_arm * RATE_HZ)

        # 起点：相对原点 0
        start_x, start_y, start_z = 0.0, 0.0, 0.0
        start_quat = [0.0, 0.0, 0.0, 1.0]

        # 终点：解算出的相对偏差
        dest_x, dest_y, dest_z = target_pos[0], target_pos[1], target_pos[2]
        #dest_quat = target_quat
        dest_quat = [0.0, 0.0, 0.0, 1.0]

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
        print(" -> 已平滑到达目标预备点！")

        # ----------------------------------------------------
        # 动作六：闭合夹爪或保持挂起
        # ----------------------------------------------------
        print("[流程 7/7] 正在闭合左手夹爪进行抓取...")
        close_cmd = Motor_Control()
        close_cmd.Position = 1.5
        robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)

        time.sleep(2.0)
        
        print("🎉 完整抓取任务执行完毕！")
        print(">>> 机器人已进入保持模式 (Hold Position)。随时可按 Ctrl+C 手动退出 <<<")
        


        # ----------------------------------------------------
        # 动作六：闭合夹爪或保持挂起
        # ----------------------------------------------------
        print("[流程 6/7] 控制手臂从相对零点平滑逼近目标...")
        
        # duration_arm = 4.0
        # steps_arm = int(duration_arm * RATE_HZ)

        # # 起点：相对原点 0
        # start_x, start_y, start_z = target_pos[0], target_pos[1], target_pos[2]
        # start_quat = [0.0, 0.0, 0.0, 1.0]

        # # 终点：解算出的相对偏差
        # dest_x, dest_y, dest_z = target_pos[0], target_pos[1] + 0.30, target_pos[2]
        # dest_quat = target_quat
        # dest_quat = [0.0, 0.0, 0.0, 1.0]

        # key_rots = R.from_quat([start_quat, dest_quat])
        # slerp = Slerp([0, 1], key_rots)

        # for i in range(1, steps_arm + 1):
        #     ratio = i / steps_arm

        #     x = start_x + (dest_x - start_x) * ratio
        #     y = start_y + (dest_y - start_y) * ratio
        #     z = start_z + (dest_z - start_z) * ratio

        #     interp_quat = slerp(ratio).as_quat()

        #     target_end_pose = ArmEndPose()
        #     target_end_pose.position = [x, y, z]
        #     target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        #     robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
        #     time.sleep(DT)

        time.sleep(0.5)
        print(" -> 已平滑到达目标预备点！")




        # 挂起主线程，保持最后姿态，等待手动 Ctrl+C 退出
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备执行安全下线流程...")

    finally:
        print("[任务结束] 回收机器人控制权...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()

if __name__ == "__main__":
    main()