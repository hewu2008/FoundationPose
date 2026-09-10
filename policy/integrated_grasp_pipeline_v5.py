import os
import sys
import time
import cv2
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
# 直接复用 zerith_client.py 中的客户端及完整处理函数。
# 客户端代码保持不变；Integrated 只负责调用并消费输出位姿。
# ---------------------------------------------------------
from zerith.zerith_client import (
    create_client,
    detect_parts,
    process_label,
    save_detection_debug,
)

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

# FoundationPose & Zerith 配置
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
ZMQ_SERVER_ADDR = "tcp://172.31.200.245:5555"
CLIENT_DEBUG_DIR = "./client_debug"
REGISTER_ITERATIONS = 5

# zerith_client 会把每个检测实例保存为：
# client_debug/<category_id>_<instance_index>/ob_in_cam/0.txt
#
# 根据当前服务端类别映射，默认：
#   左手抓取 cat2 的第 0 个实例
#   右手抓取 cat4 的第 0 个实例
# 若服务端类别映射不同，只需修改下面四个值，不需要改检测逻辑。
LEFT_TARGET_CATEGORY_ID = "cat3"
LEFT_TARGET_INSTANCE_INDEX = [0, 1, 2]
RIGHT_TARGET_CATEGORY_ID = "cat2"
RIGHT_TARGET_INSTANCE_INDEX = [0, 1, 2]

K_COLOR = np.array([
    [607.62, 0.00, 329.68],
    [0.00, 608.40, 243.36],
    [0.00, 0.00, 1.00],
], dtype=np.float64)
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

    # 动作 2：头部俯视
    # print(" -> 2/2 头部云台俯视中...")
    # head_pitch = Motor_Control()
    # head_pitch.Position = 0.0
    # target_pitch_angle = -0.2  
    # head_steps = int(1.5 * RATE_HZ) 
    
    # for i in range(1, head_steps + 1):
    #     head_pitch.Position = target_pitch_angle * (i / head_steps)
    #     robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
    #     time.sleep(DT)
        
    # time.sleep(2.0)
    # print(" -> 观测姿态调整完毕。")

# def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
#     print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
#     duration_arm = 4.0
#     steps_arm = int(duration_arm * RATE_HZ)

#     start_x, start_y, start_z = cur_xyz[0], cur_xyz[1], cur_xyz[2]
#     start_quat = cur_quat

#     dest_x, dest_y, dest_z = dest_xyz[0], dest_xyz[1], dest_xyz[2] 

#     key_rots = R.from_quat([start_quat, dest_quat])
#     slerp = Slerp([0, 1], key_rots)

#     for i in range(1, steps_arm + 1):
#         ratio = i / steps_arm

#         x = start_x + (dest_x - start_x) * ratio
#         y = start_y + (dest_y - start_y) * ratio
#         z = start_z + (dest_z - start_z) * ratio

#         interp_quat = slerp(ratio).as_quat()

#         target_end_pose = ArmEndPose()
#         target_end_pose.position = [x, y, z]
#         target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

#         robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
#         robot.setArm_high(ArmAction.RIGHT_ARM, target_end_pose)
#         time.sleep(DT)

#     time.sleep(0.5)
#     print(" -> 已平滑到达目标点！")

def _move_arm(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
              duration=4.0, rate=RATE_HZ, dt=DT):
    """单条手臂的平滑插补，供线程调用"""
    steps = int(duration * rate)
    sx, sy, sz = start_xyz
    dx, dy, dz = dest_xyz

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps + 1):
        ratio = i / steps
        x = sx + (dx - sx) * ratio
        y = sy + (dy - sy) * ratio
        z = sz + (dz - sz) * ratio
        quat = slerp(ratio).as_quat()

        pose = ArmEndPose()
        pose.position = [x, y, z]
        pose.rotation = [quat[0], quat[1], quat[2], quat[3]]

        robot.setArm_high(arm, pose)
        time.sleep(dt)


def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[流程 E] 控制双臂从相对零点同时平滑逼近目标...")

    # 左右臂不同的目标位置（y 互为相反数）
    left_dest = [dest_xyz[0],  dest_xyz[1], dest_xyz[2]]
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]   # y 取反

    # 创建线程：左臂与右臂同时运动
    t_left = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.LEFT_ARM, cur_xyz, cur_quat,
              left_dest, dest_quat)
    )
    t_right = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.RIGHT_ARM, cur_xyz, cur_quat,
              right_dest, dest_quat)
    )

    t_left.start()
    t_right.start()

    # 等待两个线程都结束
    t_left.join()
    t_right.join()

    time.sleep(0.5)
    print(" -> 双臂已同时平滑到达各自目标！")

def arm_move_rec(robot, arm, dx, dy, dz):

    duration_arm = 4.0
    steps_arm = int(duration_arm * RATE_HZ)

    ok_arm, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    start_x, start_y, start_z = arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]
    start_quat = arm_quat_rel

    # dest_x, dest_y, dest_z = arm_pos_rel[0], arm_pos_rel[1]+0.2, arm_pos_rel[2]
    dest_x, dest_y, dest_z = arm_pos_rel[0]+dx, arm_pos_rel[1]+dy, arm_pos_rel[2]+dz
    dest_quat = arm_quat_rel
    #dest_quat = target_quat

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

        robot.setArm_high(arm, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_left(robot, target_pos, target_quat):

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]]
    temp_quat = [0.0000, 0.0, 0.0000, 1]
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat)

    time.sleep(1.0)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    dest_xyz = [arm_pos_rel[0]+0.05, arm_pos_rel[1], arm_pos_rel[2]]
    dest_quat = arm_quat_rel

    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, dest_xyz, dest_quat, 3)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_right(robot, target_pos, target_quat):
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]]
    temp_quat = [0.0000, 0.0, 0.0000, 1]
    # temp_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat)

    time.sleep(1.0)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    # dest_x, dest_y, dest_z = arm_pos_rel[0]+target_pos[0]-0.09, arm_pos_rel[1]+target_pos[1]+0.04, arm_pos_rel[2]+target_pos[2]-0.02
    dest_xyz = [arm_pos_rel[0]+0.05, arm_pos_rel[1], arm_pos_rel[2]]
    dest_quat = arm_quat_rel
    # dest_quat = target_quat

    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, dest_xyz, dest_quat)

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

# ================= 3. 感知端逻辑（处理所有检测物体） =================
def build_pose_path(debug_dir, category_id, instance_index):
    """构造 zerith_client 保存的单个物体位姿文件路径。"""
    return os.path.join(
        debug_dir,
        f"{category_id}_{instance_index}",
        "ob_in_cam",
        "0.txt",
    )


def run_perception_client(rgb_path, depth_path, debug_dir=CLIENT_DEBUG_DIR):
    """
    调用 zerith_client 的原有处理流程，对 Detection 返回的所有物体逐一注册。

    不根据 label 做筛选，也不额外写根目录下的 0.txt。每个成功物体的位姿由
    zerith_client.process_label 保存到：
        <debug_dir>/<category_id>_<instance_index>/ob_in_cam/0.txt

    Returns:
        dict[(str, int), str]:
            {(category_id, instance_index): pose_file_path}
            只包含本轮成功注册并且位姿文件确实存在的物体。
    """
    print("\n[流程 C] 请求 Zerith 对所有检测物体执行 Detection + Register...")
    os.makedirs(debug_dir, exist_ok=True)

    client = create_client(ZMQ_SERVER_ADDR)
    if client is None:
        print(" -> [错误] Zerith 服务端连接失败。")
        return {}

    pose_files = {}

    try:
        # 1. Detection：直接使用 zerith_client.detect_parts，保留全部检测结果。
        color, boxes = detect_parts(client, rgb_path)
        if color is None or boxes is None:
            print(" -> [错误] Detection 调用失败。")
            return {}

        if len(boxes) == 0:
            print(" -> [警告] 当前画面未检测到任何物体。")
            return {}

        save_detection_debug(debug_dir, color, boxes)
        print(f" -> Detection 返回 {len(boxes)} 个物体，开始逐个 Register。")

        # 2. 读取与 zerith_client.main 相同的深度数据。
        depth = np.load(depth_path)

        # 同一 category_id 可能出现多个实例，编号规则与 zerith_client.main 一致。
        category_counts = {}

        for detection_index, box_dict in enumerate(boxes):
            label = box_dict["label"]
            category_id = str(box_dict["category_id"])

            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1

            box = [
                int(box_dict["x1"]),
                int(box_dict["y1"]),
                int(box_dict["x2"]),
                int(box_dict["y2"]),
            ]

            label_output_dir = os.path.join(
                debug_dir,
                f"{category_id}_{instance_index}",
            )

            print(
                f" -> [{detection_index + 1}/{len(boxes)}] "
                f"处理 {category_id}_{instance_index}: {label}"
            )

            # process_label 内部会调用 client.register，并将结果保存到
            # <label_output_dir>/ob_in_cam/0.txt。
            success = process_label(
                client=client,
                K=K_COLOR,
                color=color,
                depth=depth,
                label=label,
                category_id=category_id,
                box=box,
                mesh_bbox=None,
                to_origin=None,
                label_output_dir=label_output_dir,
                register_iterations=REGISTER_ITERATIONS,
                show=False,
            )

            if not success:
                print(f" -> [警告] {category_id}_{instance_index} 注册失败，继续处理其他物体。")
                continue

            pose_path = build_pose_path(
                debug_dir,
                category_id,
                instance_index,
            )

            if not os.path.isfile(pose_path):
                print(f" -> [警告] 注册返回成功，但未找到位姿文件: {pose_path}")
                continue

            pose_files[(category_id, instance_index)] = pose_path
            print(f" -> 位姿已保存: {pose_path}")

        print(f"\n -> 本轮共成功获得 {len(pose_files)} 个物体位姿：")
        for (category_id, instance_index), pose_path in pose_files.items():
            print(f"    {category_id}_{instance_index}: {pose_path}")
        for k in category_counts:
            if k == LEFT_TARGET_CATEGORY_ID:
                LEFT_TARGET_INSTANCE_INDEX = [i for i in range(category_counts[k])]
            elif k == RIGHT_TARGET_CATEGORY_ID:
                RIGHT_TARGET_INSTANCE_INDEX = [i for i in range(category_counts[k])]

        return pose_files

    except Exception as e:
        print(f" -> [异常] 感知端运行崩溃: {e}")
        return {}

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

def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, flag):
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

    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T_comp = np.eye(4)
    
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

def rotate_90_degrees(robot: H1Robot, direction="left"):
    DT = 0.2
    ANGULAR_SPEED = 0.5  
    TARGET_ANGLE = 1.5708  
    DURATION = TARGET_ANGLE / ANGULAR_SPEED + 1.2  
    
    if direction == "left":
        yaw_speed = ANGULAR_SPEED  
    else:
        yaw_speed = -ANGULAR_SPEED  
    
    print(f"[Chassis] 旋转90度{direction}, 角速度{ANGULAR_SPEED}rad/s, 持续{DURATION:.2f}s")
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            robot.setChassis_high(0.0, yaw_speed)
            
            elapsed = time.time() - start_time
            rotated_degrees = (ANGULAR_SPEED * elapsed) * 180 / 3.14159
            print(f"[Chassis] 已旋转: {rotated_degrees:.1f}°")
            
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time = DT - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 旋转90度完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 中断")
        robot.setChassis_high(0.0, 0.0)

# ================= 主控制流 =================

def grasp_by_left(robot, pose_files, index):
    # ------------------------------------------------
    # 1. 机器人姿态准备
    # ------------------------------------------------
    
    left_pose_path = pose_files.get(
        (LEFT_TARGET_CATEGORY_ID, index)
    )
    if left_pose_path is None:
        print(
            "[致命错误] 本轮没有生成左手目标位姿："
            f"{LEFT_TARGET_CATEGORY_ID}_{LEFT_TARGET_INSTANCE_INDEX}，终止左手抓取！"
        )
        return

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
        return

    T_obj_cam = load_pose_matrix(left_pose_path)
    target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, 0
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

    arm_move_rec(robot, ArmAction.LEFT_ARM, -0.2, 0, 0)
    prepare_robot_posture(robot, 0.67, 1.2, 0.75, 1.0)
    arm_move_rec(robot, ArmAction.LEFT_ARM, 0, 0.25, 0)
    #arm_move_rec(robot, ArmAction.LEFT_ARM, 0.2, 0, -0.1)

    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)


    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.02, 0.30], [0, 0, 0, 1])
    time.sleep(1.0)
    prepare_robot_posture(robot, 0.75, 1.0, 0.67, 1.2)
    time.sleep(1.0)


def grasp_by_right(robot, pose_files, index):
    right_pose_path = pose_files.get(
        (RIGHT_TARGET_CATEGORY_ID, index)
    )
    if right_pose_path is None:
        print(
            "[致命错误] 本轮没有生成右手目标位姿："
            f"{RIGHT_TARGET_CATEGORY_ID}_{RIGHT_TARGET_INSTANCE_INDEX}，终止右手抓取！"
        )
        return
    
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position",None)
    cam_quat_rel = getattr(cam_state, "rotation", None)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    if not (ok_cam and ok_arm):
        print("传感器位姿获取失败！")
        return
    
    #位姿转换矩阵计算
    T_obj_cam = load_pose_matrix(right_pose_path)
    target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, 1
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
    prepare_robot_posture(robot, 0.67, 1.2, 0.75, 1.0)
    arm_move_rec(robot, ArmAction.RIGHT_ARM, 0, -0.25, 0)
    #arm_move_rec(robot, ArmAction.RIGHT_ARM, 0.2, 0, -0.1)

    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)
    
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.02, 0.30], [0, 0, 0, 1])
    time.sleep(1.0)
    prepare_robot_posture(robot, 0.75, 1.0, 0.67, 1.2)
    time.sleep(1.0)

def main():
    robot = H1Robot()
    try:
        print("[INIT] 实例化机器人并连接...")
        if not robot.robot_connect():
            print("连接机器人失败！")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        time.sleep(1.5)
        prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
        arm_move_pre(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.02, 0.30],
                [0.0, 0.0, 0.0, 1.0])
        #chassis_move(robot)
        #prepare_robot_posture(robot, 0.6, 1.0, 0.5, 1.0)
        # time.sleep(2.0)

        # # ------------------------------------------------
        # # 2. 拍摄 RGB-D 照片
        # # ------------------------------------------------
        # rgb_path, depth_path = capture_rgbd_data()
    
        # # ------------------------------------------------
        # # 3. 对当前画面中的所有检测物体执行注册并保存各自位姿
        # # ------------------------------------------------
        # pose_files = run_perception_client(rgb_path, depth_path)

        # for index in LEFT_TARGET_INSTANCE_INDEX:
        #     grasp_by_left(robot, pose_files, index)
        #     time.sleep(2.0)

        # for index in RIGHT_TARGET_INSTANCE_INDEX:
        #     grasp_by_right(robot, pose_files, index)
        #     time.sleep(2.0)

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
