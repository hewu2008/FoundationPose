import os
import sys
import time
import cv2
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import threading

# ================= 导入 SDK 及环境配置 =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

# 机器人 SDK
from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode, EtherCAT_Motor_Index,
    ArmAction, ArmPose, ArmEndPose, Motor_Control
)

# 相机 gRPC 客户端
from camera_client import CameraClient

# ---------------------------------------------------------
# 新增：从 zerith_client 导入 ZMQ 客户端及辅助函数
# ---------------------------------------------------------
from FoundationPose.zerith.zerith_client_pre import ZerithFoundationPoseClient, detect_parts

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

# FoundationPose & Zerith 配置
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
ZMQ_SERVER_ADDR = "tcp://172.31.200.250:5555"
MESH_FILE = "model/DPPUB-204001196-AAX_01_01.obj"
TARGET_LABELS0 = ["a white translucent plastic brake fluid reservoir with a blue or black cap"]
## TARGET_LABELS = ["a T-shaped black metal car door checker with a wide top head and a narrow bottom stem"]
TARGET_LABELS1 = ["a white smooth solid rectangular or square block with straight edges, with or without holes"]
# ================= 1. 机器人姿态准备模块 =================
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



def _move_arm(robot, arm, dest_xyz, dest_quat):
    """单条手臂的平滑插补，供线程调用"""
    target_pose = ArmEndPose()
    target_pose.position = dest_xyz
    target_pose.rotation = dest_quat
    robot.setArmMove_high(arm, target_pose)
    time.sleep(2.0)


def arm_move_pre(robot, dest_xyz, dest_quat):
    print("\n[流程 E] 控制双臂从相对零点同时平滑逼近目标...")

    # 左右臂不同的目标位置（y 互为相反数）
    left_dest = dest_xyz
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]   # y 取反

    _move_arm(robot, ArmAction.LEFT_ARM, left_dest, dest_quat)
    time.sleep(2.0)
    _move_arm(robot, ArmAction.RIGHT_ARM, right_dest, dest_quat)

    time.sleep(0.5)
    print(" -> 双臂已同时平滑到达各自目标！")

def arm_move_rec(robot, arm, dx, dy, dz):
    ok_arm, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    dest_xyz = [arm_pos_rel[0]+dx, arm_pos_rel[1]+dy, arm_pos_rel[2]+dz]
    dest_quat = arm_quat_rel
    
    _move_arm(robot, arm, dest_xyz, dest_quat)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_left(robot, target_pos, target_quat):
    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [target_pos[0]-0.20, target_pos[1]+0.05, target_pos[2]-0.01]
    temp_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    _move_arm(robot, ArmAction.LEFT_ARM, temp_xyz, temp_quat)

    # dest_xyz = [arm_pos_rel[0]+target_pos[0]-0.11, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]-0.01]
    # #dest_x, dest_y, dest_z = arm_pos_rel[0]+target_pos[0], arm_pos_rel[1]+target_pos[1], arm_pos_rel[2]+target_pos[2]
    # dest_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    # # dest_quat = target_quat
    time.sleep(2.0)
    dest_xyz = temp_xyz
    dest_xyz[0] += 0.1
    dest_quat = [0.0000, 0.3827, 0.0000, 0.9239]

    _move_arm(robot, ArmAction.LEFT_ARM, dest_xyz, dest_quat)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_right(robot, target_pos, target_quat):
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")

    dest_xyz = target_pos[0], target_pos[1]+0.02, target_pos[2]
    dest_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    # dest_quat = target_quat

    _move_arm(robot, ArmAction.RIGHT_ARM, dest_xyz, dest_quat)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

# ================= 2. RGB-D 获取模块 =================
def capture_rgbd_data():
    """获取 gRPC 图像并保存"""
    print(f"\n[流程 B] 正在连接相机服务获取 RGB-D 数据 ({GRPC_TARGET})...")
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    client.start()
    
    rgb_path = "zerith_rgb.png"
    depth_path = "zerith_depth.npy"
    
    try:
        max_retries = 50
        for i in range(max_retries):
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME) 
            
            if depth_data is not None and color_data is not None:
                depth_raw_mm, _ = depth_data
                color_raw, _ = color_data
                
                # 转换为算法需要的 float32 (米)
                depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0
                
                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_raw_m)
                print(f" -> 捕获成功！已保存 {rgb_path} 和 {depth_path}")
                return rgb_path, depth_path
                
            time.sleep(0.1)
        raise TimeoutError("获取图像超时！请检查 gRPC 节点。")
    finally:
        client.stop()

# ================= 3. 感知端逻辑 (Detection + Register + Track) =================
def run_perception_client(rgb_path, depth_path, mesh_file, target_labels):
    """调用 zerith_client 中的模块完成感知推理"""
    print("\n[流程 C] 请求 Zerith FoundationPose 推理位姿 (Detection + 1 Register + 9 Track)...")
    
    try:
        mesh = trimesh.load(mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        print(f" -> 已加载 Mesh: {mesh_file}")
    except Exception as e:
        print(f" -> [警告] Mesh 加载失败 (不影响推理): {e}")

    # 直接使用 zerith_client 导入的客户端
    client = ZerithFoundationPoseClient(ZMQ_SERVER_ADDR)
    
    response = client.ping()
    if response['status'] != 'success':
        print(f" -> [错误] 服务端连接失败: {response.get('message', '未知错误')}")
        return False
    print(" -> 服务端连接成功！开始推理...")
    
    try:
        # 1. 目标检测 (Detection)
        print(" -> 正在进行目标检测提取 Bounding Box...")
        target_label_str = target_labels[0] if isinstance(target_labels, list) else target_labels
        target_box = None
        
        # 调用 zerith_client 的 detect_parts 方法，返回 RGB 图像和检测框
        color, boxes = detect_parts(client, rgb_path)

        if boxes:
            for box_dict in boxes:
                # 若检测到的 label 包含目标 label 或者完全一致，则提取框
                if box_dict['label'] == target_label_str or target_label_str in box_dict['label']:
                    target_box = [int(box_dict['x1']), int(box_dict['y1']), int(box_dict['x2']), int(box_dict['y2'])]
                    target_label_str = box_dict['label']  # 修正为算法实际返回的名称
                    print(f" -> 🎯 找到目标 '{target_label_str}', Box: {target_box}")
                    break
            
            if target_box is None:
                print(f" -> [警告] 视野内未匹配到对应标签 ({target_label_str})，将尝试无框注册。")
        else:
            print(" -> [警告] Detection 未返回任何框，将尝试盲注册。")

        # 2. 注册与追踪 (Register + Track)
        depth = np.load(depth_path)
        K_color = np.array([
            [607.62, 0.00,  329.68],
            [0.00,  608.40, 243.36],
            [0.00,  0.00, 1.00]
        ])
        
        final_pose = None
        for i in range(10):
            if i == 0:
                print(f" -> [第 {i} 帧] 正在进行注册 (Registering)...")
                # 适配新版 register 接口：传入 label 和 box
                response = client.register(
                    K=K_color, rgb=color, depth=depth,
                    label=target_label_str, box=target_box, threshold=0.41, iteration=5
                )
                if response['status'] != 'success':
                    print(f" -> [错误] 注册失败: {response.get('message', '未知错误')}")
                    return False
                pose = response['pose']
                if hasattr(pose, 'numpy'): pose = pose.numpy()
                print(" -> 注册成功！")
            else:
                start_time = time.time()
                response = client.track(
                    K=K_color, rgb=color, depth=depth, iteration=2
                )
                if response['status'] != 'success':
                    print(f" -> [错误] 跟踪失败: {response.get('message', '未知错误')}")
                    return False
                pose = response['pose']
                if hasattr(pose, 'numpy'): pose = pose.numpy()
                end_time = time.time()
                print(f" -> [第 {i} 帧] 跟踪成功，耗时: {end_time - start_time:.4f}s")
            
            final_pose = pose

        if final_pose is not None:
            np.savetxt("0.txt", final_pose.reshape(4, 4))
            print("\n -> 🎉 感知推理(Detection+Register+Track)全部完成！已将最终 4x4 矩阵写入 0.txt。")
            return True
            
        return False

    except Exception as e:
        print(f" -> [异常] 感知端运行崩溃: {e}")
        return False
        
    finally:
        client.close()

# ================= 4. 抓取运算与执行模块 =================
def load_pose_matrix(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row: matrix.append(row)
    return np.array(matrix)

def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, T_obj_cam, flag):
    """坐标系变换：感知位姿 -> 手臂相对零点目标位姿"""
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    if flag == 0:
        T3[:3, 3] = [-0.5743, -0.1800, -0.1208]
    else:
        T3[:3, 3] = [-0.5743, 0.1800, -0.1208]


    T_obj_in_arm = np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam)))

    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] 

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat


# ================= 主控制流 =================

def grasp_by_left(robot):
    # ------------------------------------------------
    # 1. 机器人姿态准备
    # ------------------------------------------------
    time.sleep(1.5)
    prepare_robot_posture(robot, 0, 0, 0.6, 1.0)
    arm_move_pre(robot, [0.0, 0.02, 0.30], [0.0, 0.0, 0.0, 1.0])
    prepare_robot_posture(robot, 0.6, 1.0, 0.35, 1.0)
    time.sleep(1.5)
    
    # ------------------------------------------------
    # 2. 拍摄 RGB-D 照片
    # ------------------------------------------------
    rgb_path, depth_path = capture_rgbd_data()

    # ------------------------------------------------
    # 3. 运行更新版感知逻辑并保存 0.txt
    # ------------------------------------------------
    success = run_perception_client(rgb_path, depth_path, MESH_FILE, TARGET_LABELS0)
    if not success:
        print("[致命错误] 感知位姿获取失败，终止抓取流程！")
        return

    # ------------------------------------------------
    # 4. 读取手臂/相机的当前状态并解算最终目标点
    # ------------------------------------------------
    print("\n[流程 D] 读取位姿状态并进行矩阵解算...")
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None) 
    cam_quat_rel = getattr(cam_state, "rotation", None) 

    if not ok_cam:
        print("传感器位姿获取失败！")
        return

    T_obj_cam = load_pose_matrix("0.txt")
    target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, T_obj_cam, 0
    )
    print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    # ------------------------------------------------
    # 5. 执行手臂平移逼近目标
    # ------------------------------------------------
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    arm_move_left(robot, target_pos, target_quat)

    # ------------------------------------------------
    # 6. 闭合夹爪进行抓取
    # ------------------------------------------------
    print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)

    time.sleep(2.0)

    # arm_move_rec(robot, ArmAction.LEFT_ARM, -0.2, 0, 0)
    # prepare_robot_posture(robot, 0.35, 1.0, 0.45, 0.8)
    # arm_move_rec(robot, ArmAction.LEFT_ARM, 0.2, 0.40, -0.1)

    # close_cmd.Position = 0.0
    # robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)

    # _move_arm(robot, ArmAction.LEFT_ARM, [0.0, 0.02, 0.25], [0, 0, 0, 1])
    # time.sleep(1.0)
    # prepare_robot_posture(robot, 0.45, 0.8, 0.35, 1.0)


def grasp_by_right(robot):
    #获取图片
    rgb_path, depth_path = capture_rgbd_data()
    
    #foundationpose获取位姿
    success = run_perception_client(rgb_path, depth_path, MESH_FILE, TARGET_LABELS1)
    if not success:
        print("[致命错误] 感知位姿获取失败，终止抓取流程！")
        return
    
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position",None)
    cam_quat_rel = getattr(cam_state, "rotation", None)


    if not ok_cam :
        print("传感器位姿获取失败！")
        return
    
    #位姿转换矩阵计算
    T_obj_cam = load_pose_matrix("0.txt")
    target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, T_obj_cam, 1
    )
    print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    #grasp
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    arm_move_right(robot, target_pos, target_quat)

    print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)

    time.sleep(2.0)

    arm_move_rec(robot, ArmAction.RIGHT_ARM, -0.2, 0, 0)
    prepare_robot_posture(robot, 0.35, 1.0, 0.45, 0.8)
    arm_move_rec(robot, ArmAction.RIGHT_ARM, 0.2, -0.40, -0.1)

    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)
    
    _move_arm(robot, ArmAction.RIGHT_ARM, [0.0, 0.02, 0.25], [0, 0, 0, 1])
    time.sleep(1.0)
    #prepare_robot_posture(robot, 0.40, 0.8, 0.35, 1.0)

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

        grasp_by_left(robot)
        #grasp_by_right(robot)
       
        while True:
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
