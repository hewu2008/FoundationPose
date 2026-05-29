from Release.linux import xCoreSDK_python
import sys
import os
import argparse
import time
import glob
import numpy as np
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R

# 添加路径 (请确保这些路径在你的机器上是正确的)
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

from ultralytics import YOLO
# 本地模块导入
from estimater import *
from ultralytics import SAM
from cam_2_base_transform import *
import pyrealsense2 as rs
from groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict
import groundingdino.datasets.transforms as T
from PIL import Image
import torch
from pinocchio_rokae import PinocchioJaka
# --- FoundationPose Monkey Patching ---
original_init = FoundationPose.__init__
original_register = FoundationPose.register


def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None,
                  debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False


def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True
    return pose


FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()


class PoseEstimatorApp:
    def __init__(self, mesh_path, text_prompt="box ."):
        self.mesh_path = mesh_path
        self.text_prompt = text_prompt  # Grounding DINO 的提示词
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_model = dino_load_model(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth"
        ).to(self.device)

        self.pose_est_instance = None
        self.is_initialized = False

        right_ip = "192.168.2.160"
        left_ip = "192.168.2.161"
        local_ip = "192.168.2.51"
        # self.right_arm = xCoreSDK_python.xMateErProRobot(right_ip, local_ip)
        # self.right_ec = {}
        # self.right_arm.setOperateMode(xCoreSDK_python.OperateMode.automatic, self.right_ec)
        # self.right_arm.setPowerState(True, self.right_ec)
        #
        # self.left_arm = xCoreSDK_python.xMateErProRobot(left_ip, local_ip)
        # self.left_ec = {}
        # self.left_arm.setOperateMode(xCoreSDK_python.OperateMode.automatic, self.left_ec)
        # self.left_arm.setPowerState(True, self.left_ec)

        self.mesh_path = mesh_path
        # self.target_coords = (target_x, target_y)

        # --- RealSense SDK Initialization ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        self.cam_K = self.get_intrinsics()

        # 加载单个指定的 Mesh
        print(f"Loading mesh: {self.mesh_path}")
        self.mesh = trimesh.load(self.mesh_path)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        # 初始化 FoundationPose 组件
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # 初始化 SAM2
        self.seg_model = SAM("sam2.1_b.pt")

        self.pose_est_instance = None  # 存储 FoundationPose 实例
        self.is_initialized = False  # 是否已成功根据坐标定位物体
        self.i = 0
        # 右臂手眼标定结果
        self.H_base_right = np.eye(4)
        self.H_base_right[:3, :3] = np.array(
            [[0.07651209 , 0.8993466,   0.43048995],
             [0.07671232, - 0.43578554,  0.89677544],
            [0.99411323, - 0.03559028, - 0.10233382],
        ])
        self.H_base_right[:3, 3] = np.array([-0.095270735, 0.03396315851, -0.16102384115])
        # 左臂手眼标定结果
        self.H_base_left = np.eye(4)
        self.H_base_left[:3, :3] = np.array(
            [[ 0.05127495,  0.88721808,  0.45849204],
                [-0.02531944,  0.46010325, -0.88750433],
                [-0.99836356,  0.03389798,  0.04605563]])
        self.H_base_left[:3, 3] = np.array([-0.08070737382, -0.04238701972, -0.08477412667])

        self.ik_left = PinocchioJaka("./ar5l_description/urdf/AR5-5_07L-W4S4A2.urdf")
        self.ik_right = PinocchioJaka("./ar5r_description/urdf/AR5-5_07R-W4S4A2.urdf")
        # self.pose_est_instance = FoundationPose(
        #     model_pts=self.mesh.vertices,
        #     model_normals=self.mesh.vertex_normals,
        #     mesh=self.mesh,
        #     scorer=self.scorer,
        #     refiner=self.refiner,
        #     glctx=self.glctx
        # )

    def get_intrinsics(self):
        video_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_stream.get_intrinsics()
        return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                depth_image = depth_data.astype(np.float32) * depth_scale

                color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # results = self.model(color_image, conf=0.5, verbose=False)


                self.process_frame(color_rgb, depth_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()

    # --- 新增：将相机 RGB 帧转换为 DINO 需要的 Tensor ---
    def transform_image_for_dino(self, cv2_rgb_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil_image = Image.fromarray(cv2_rgb_image)
        image_transformed, _ = transform(pil_image, None)
        return image_transformed

    def process_frame(self, color_image, depth_image):
        H, W = color_image.shape[:2]
        depth_image[(depth_image < 0.1) | (depth_image >= np.inf)] = 0

        visualization_image = np.copy(color_image)

        # 1. 尝试初始化 (仅执行一次，直到找到目标坐标下的物体)
        if not self.is_initialized:

            dino_tensor = self.transform_image_for_dino(color_image).to(self.device)
            boxes, logits, phrases = dino_predict(
                model=self.dino_model,
                image=dino_tensor,
                caption=self.text_prompt,
                box_threshold=0.35,
                text_threshold=0.25
            )
            target_mask = None

            if len(boxes) > 0:
                # 1. 将 DINO 的 [cx, cy, w, h] (0~1) 转换为像素坐标 [xmin, ymin, xmax, ymax]
                # boxes 形状为 [N, 4]
                boxes_np = boxes.cpu().numpy()
                boxes_xyxy = []
                for box in boxes_np:
                    cx, cy, w, h = box
                    xmin, ymin = (cx - w / 2) * W, (cy - h / 2) * H
                    xmax, ymax = (cx + w / 2) * W, (cy + h / 2) * H
                    boxes_xyxy.append([xmin, ymin, xmax, ymax])

                boxes_xyxy = np.array(boxes_xyxy)

                # 2. 调用优化后的决策函数 (传入深度图 depth_map 和 图像尺寸)
                # 优先：深度近 > 图像靠上(ymin小) > 水平居中
                optimal_idx = self.select_optimal_box(boxes_xyxy, depth_image, (H, W))

                if optimal_idx is not None:
                    # 提取最终选定的目标框参数
                    xmin, ymin, xmax, ymax = map(int, boxes_xyxy[optimal_idx])

                # B. 将 DINO 的边界框作为 Prompt 喂给 SAM2
                sam_res = self.seg_model.predict(
                    color_image,
                    bboxes=[xmin, ymin, xmax, ymax],
                    verbose=False
                )[0]

                if sam_res.masks is not None:
                    # 获取 SAM2 生成的 Mask (布尔型转为 uint8)
                    mask_data = sam_res.masks.data[0].cpu().numpy()
                    target_mask = (mask_data * 255).astype(np.uint8)
                    print("SAM2 generated mask successfully.")
            sam_vis_disk = cv2.imread("sam.jpg")
            # 这里的逻辑是：sam.jpg 中绿色区域（我们之前画的 mask_overlay）就是掩码
            # 或者更简单的做法：直接保存和读取二值化的 mask 结果

            # 提取绿色通道作为掩码的参考（假设 mask_overlay 是绿色的）
            # 更好的做法是直接读取我们保存时的二进制掩码，但如果只有可视化图：
            hsv = cv2.cvtColor(sam_vis_disk, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 43, 46])
            upper_green = np.array([77, 255, 255])
            target_mask = cv2.inRange(hsv, lower_green, upper_green)

            if target_mask is not None:
                # 创建 FoundationPose 实例并进行首次 Register
                self.pose_est_instance = FoundationPose(
                    model_pts=self.mesh.vertices,
                    model_normals=self.mesh.vertex_normals,
                    mesh=self.mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    glctx=self.glctx
                )
                start_time = time.time()
                # 首次注册位姿
                _ = self.pose_est_instance.register(
                    K=self.cam_K, rgb=color_image, depth=depth_image,
                    ob_mask=target_mask, iteration=args.est_refine_iter
                )
                self.is_initialized = True
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("[INFO] elapsed time: {:.2f}s".format(elapsed_time))
            else:
                # 在窗口上画个准星
                # cv2.drawMarker(visualization_image, self.target_coords, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(visualization_image, "Waiting for object at target coords...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. 实时跟踪逻辑
        elif self.is_initialized:
            start_time2 = time.time()
            # 持续跟踪
            pose = self.pose_est_instance.track_one(
                rgb=color_image, depth=depth_image, K=self.cam_K, iteration=args.track_refine_iter
            )
            center_pose = pose @ np.linalg.inv(self.to_origin)
            # center_pose = pose
            # 绘制 3D Box 和 坐标轴
            visualization_image = draw_posed_3d_box(self.cam_K, img=visualization_image, ob_in_cam=center_pose,
                                                    bbox=self.bbox)
            visualization_image = draw_xyz_axis(visualization_image, ob_in_cam=center_pose, scale=0.1, K=self.cam_K,
                                                thickness=3, transparency=0, is_input_rgb=True)

            # (可选) 调用机器人逻辑
            # self.print_pose(center_pose, 0)
            end_time2 = time.time()
            elapsed_time2 = end_time2 - start_time2
            print("[INFO2] elapsed time: {:.2f}s".format(elapsed_time2))

        # 显示结果
        cv2.imshow('Real-time Tracking', visualization_image[..., ::-1])

    def print_pose(self, center_pose, idx):
            current_right_joint = self.right_arm.jointPos(self.right_ec)
            current_left_joint = self.left_arm.jointPos(self.left_ec)
            pose_cam_mm = center_pose.copy()

            ss_rpy = R.from_matrix(pose_cam_mm[:3, :3]).as_euler('xyz', degrees=False)
            pose_cam_mm[:3, 3] = pose_cam_mm[:3, 3]
            # y朝下，z朝前，x朝右
            # 3. 矩阵乘法: (Base <- Cam) * (Cam <- Obj) = (Base <- Obj)
            pose_base_right = self.H_base_right @ pose_cam_mm
            position_right_base = pose_base_right[:3, 3]
            position_right_base[2] += 0.5
            # position_right_base[2] += 0.35
            position_right_base[1] -= 0.12
            position_right_base[0] -= 0.07

            pose_base_left = self.H_base_left @ pose_cam_mm
            position_left_base = pose_base_left[:3, 3]
            position_left_base[2] += 0.5
            # position_left_base[2] += 0.35
            position_left_base[1] += 0.10


            target_right_rpy = np.array([-1.57, 3.14, -1.0])
            target_left_rpy = np.array([1.57, -3.14, 1.0])
            # target_left_rpy = np.array([1.57, 3.14, 0])
            # Jaka机械臂参数
            # target_rpy = np.array([0, 1.57, 0])

            q_right_sol, right_success, right_cost = self.ik_right.ik_pinocchio(current_right_joint, position_right_base, target_right_rpy)
            q_left_sol, left_success, left_cost = self.ik_left.ik_pinocchio(current_left_joint, position_left_base, target_left_rpy)
            # 机械臂运动
            absjcmd = xCoreSDK_python.MoveAbsJCommand(q_right_sol, 100, 10)
            cmdID = xCoreSDK_python.PyString()
            self.right_arm.moveAppend([absjcmd], cmdID, self.right_ec)  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
            #
            #
            # absjcmd_left = xCoreSDK_python.MoveAbsJCommand(q_left_sol, 100, 10)
            # cmdID = xCoreSDK_python.PyString()
            # self.left_arm.moveAppend([absjcmd_left], cmdID, self.left_ec)  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
            #
            self.right_arm.moveStart(self.right_ec)
            # self.left_arm.moveStart(self.left_ec)
            # print(f"right_arm:{self.right_ec['message']}")
            # print(f"left_arm:{self.left_ec['message']}")
            # wait_robot(robot, ec)
            print("-" * 20)

    import numpy as np

    def select_optimal_box(self, boxes_xyxy, depth_image, img_shape):
        if len(boxes_xyxy) == 0:
            return None

        H, W = img_shape[:2]
        img_center_x = W / 2
        total_area = H * W

        results = []
        for i, box in enumerate(boxes_xyxy):
            xmin, ymin, xmax, ymax = map(int, box)

            # 1. 基础过滤：面积太小可能为误检
            area = (xmax - xmin) * (ymax - ymin)
            if area < (total_area * 0.01):  # 降低门槛至1%，避免漏掉高处较远的箱子
                continue

            # 2. 计算中心水平偏移 (越小越居中)
            box_center_x = (xmin + xmax) / 2
            dist_to_center_x = abs(box_center_x - img_center_x)

            # 3. 计算“最上方”指标 (ymin 越小代表在图像中越靠上)
            # 注意：在搬箱子场景中，顶层的箱子 ymin 通常更小
            top_y = ymin

            # 4. 深度提取 (ROI 逻辑优化)
            # 取箱子中心区域的深度，避免边缘噪声
            roi_x1, roi_x2 = int(xmin + (xmax - xmin) * 0.3), int(xmax - (xmax - xmin) * 0.3)
            roi_y1, roi_y2 = int(ymin + (ymax - ymin) * 0.3), int(ymax - (ymax - ymin) * 0.3)

            depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depths = depth_roi[depth_roi > 0]

            if len(valid_depths) == 0:
                avg_depth = float('inf')
            else:
                # 采用 10 分位数表示该物体最靠近相机的表面距离
                avg_depth = np.percentile(valid_depths, 10)

            results.append({
                'original_index': i,
                'top_y': top_y,
                'dist_x': dist_to_center_x,
                'depth': avg_depth,
                'area': area
            })

        if not results:
            return None

        # 提取各项指标进行归一化
        tops = np.array([r['top_y'] for r in results])
        dists_x = np.array([r['dist_x'] for r in results])
        depths = np.array([r['depth'] for r in results])

        def safe_normalize(data, reverse=False):
            d_min, d_max = data.min(), data.max()
            if d_max == d_min: return np.ones_like(data)
            norm = (data - d_min) / (d_max - d_min)
            return 1.0 - norm if reverse else norm

        # 归一化逻辑（全部转换为：分值越高越优）
        s_top = safe_normalize(tops, reverse=True)  # Y轴坐标越小，分数越高 (最上面)
        s_dist = safe_normalize(dists_x, reverse=True)  # 水平偏移越小，分数越高 (居中)

        s_depth = np.zeros_like(depths)
        valid_mask = depths != float('inf')
        if valid_mask.any():
            s_depth[valid_mask] = safe_normalize(depths[valid_mask], reverse=True)  # 深度越小，分数越高 (离得近)

        total_scores = (0.4 * s_depth) + (0.4 * s_top) + (0.2 * s_dist)

        best_idx_in_results = np.argmax(total_scores)
        return results[best_idx_in_results]['original_index']
def main():
    # 直接指定路径和坐标
    # mesh_path = "/home/jszn/wangcheng/FoundationPoseROS2/demo_data/box/small2.obj"
    mesh_path = "/home/jszn/wangcheng/FoundationPoseROS2/demo_data/box/textured_simple.obj"
    # mesh_path = "/home/jszn/wangcheng/FoundationPoseROS2/demo_data/box/big.obj"
    # text_prompt = "cardboard box ."
    text_prompt = "material box ."

    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return

    app = PoseEstimatorApp(mesh_path, text_prompt)
    app.run()


if __name__ == '__main__':
    main()