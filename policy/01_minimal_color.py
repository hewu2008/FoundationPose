"""
@file 01_minimal_color.py
@brief 【示例 01】最简彩色图获取 + 图像标定叠加
* [功能介绍]
1. 展示如何初始化 CameraClient 并建立连接。
2. 演示非阻塞式获取最新彩色图像 (RGB) 并使用 OpenCV 显示。
3. 新增「图像标定模式」：把固定路径下的标定图像半透明叠加到实时相机画面上，
   用于目视核对相机是否与标定状态一致，不透明度可实时调整。
* [使用说明]
1. 修改 GRPC_TARGET 为服务端真实的 IP 地址。
2. 修改 CAMERA_NAME 为服务端对应的相机 ID。
3. 修改 CALIB_IMAGE 为标定图像的路径（相对仓库根目录或绝对路径）。
* [快捷键]
- c : 开 / 关图像标定模式（半透明叠加）
- [ : 降低标定图不透明度
- ] : 提高标定图不透明度
- q : 退出预览
"""

import os
import sys

import cv2
import numpy as np

# --- 路径兼容处理：确保能找到上一级目录中的 camera_client.py ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
from camera_client import CameraClient

# --- 图像标定模式配置 ---
# 标定图像路径：相对路径按仓库根目录解析，也可直接填绝对路径。
CALIB_IMAGE = "assets/imgs/zero.png"
CALIB_ALPHA_DEFAULT = 0.5   # 标定图初始不透明度，范围 [0, 1]
CALIB_ALPHA_STEP = 0.05     # 每次调整不透明度的步长


def resolve_path(path):
    """相对路径按仓库根目录解析，绝对路径原样返回。"""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_calibration_image(path):
    """读取标定图像并统一为 BGR 三通道；失败返回 None。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:                       # 灰度图 -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                 # BGRA -> 先按 alpha 合成到黑底
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    else:
        img = np.ascontiguousarray(img[..., :3])
    return img


def blend_calibration(frame, calib, alpha):
    """把标定图按 alpha 半透明叠加到相机画面上。"""
    return cv2.addWeighted(frame, 1.0 - alpha, calib, alpha, 0.0)


def main():
    # ---------------------------------------------------------
    # 【用户配置区】请根据实际环境修改以下参数
    # ---------------------------------------------------------
    GRPC_TARGET = "localhost:50051"  # 服务端 IP:端口
    CAMERA_NAME = "rs/cam_high"      # 彩色相机 ID
    # ---------------------------------------------------------

    # 1. 初始化客户端
    client = CameraClient(grpc_target=GRPC_TARGET)
    window_sizes = {}

    def show_image(window_name, image):
        image_size = (image.shape[1], image.shape[0])
        if window_name not in window_sizes:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if window_sizes.get(window_name) != image_size:
            cv2.resizeWindow(window_name, image_size[0], image_size[1])
            window_sizes[window_name] = image_size
        cv2.imshow(window_name, image)

    # 2. 预读标定图像（固定路径），失败则标定模式不可用
    calib_path = resolve_path(CALIB_IMAGE)
    calib_image = load_calibration_image(calib_path)
    if calib_image is None:
        print(f"[!] 未能读取标定图像 [{calib_path}]，按 'c' 将无法开启标定模式。")
    else:
        print(f"[*] 已加载标定图像: {calib_path}")
    calib_fitted = {}                # 按画面尺寸缓存缩放后的标定图
    calib_mode = False               # 图像标定模式开关
    alpha = CALIB_ALPHA_DEFAULT      # 标定图不透明度

    try:
        # 3. 启动客户端 (执行 gRPC 握手与 WebRTC 建立)
        print(f"[*] 正在连接服务 {GRPC_TARGET} ...")
        client.start()
        print(f"[+] 连接成功。正在获取相机 [{CAMERA_NAME}] 的画面...")
        print("[i] 快捷键: c 切换标定叠加 / [ ] 调整不透明度 / q 退出")

        while True:
            # 4. 获取最新彩色帧 (返回: (numpy_bgr_array, timestamp) 或 None)
            data = client.get_latest_frame(CAMERA_NAME)

            if data:
                img, ts = data
                display = img

                # 5. 图像标定模式：把标定图半透明覆盖到当前相机画面上
                if calib_mode and calib_image is not None:
                    image_size = (img.shape[1], img.shape[0])
                    if image_size not in calib_fitted:
                        calib_fitted[image_size] = cv2.resize(
                            calib_image, image_size, interpolation=cv2.INTER_LINEAR
                        )
                    display = blend_calibration(img, calib_fitted[image_size], alpha)
                    cv2.putText(
                        display, f"CALIB  alpha={alpha:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )

                show_image("Color Viewer", display)

            # 6. 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[+] 用户请求退出。")
                break
            elif key == ord('c'):
                if calib_image is None:
                    print(f"[!] 标定图像不可用，无法开启标定模式: {calib_path}")
                else:
                    calib_mode = not calib_mode
                    print(f"[*] 图像标定模式: {'开启' if calib_mode else '关闭'}")
            elif key == ord('['):
                alpha = max(0.0, alpha - CALIB_ALPHA_STEP)
                print(f"[*] 标定图不透明度: {alpha:.2f}")
            elif key == ord(']'):
                alpha = min(1.0, alpha + CALIB_ALPHA_STEP)
                print(f"[*] 标定图不透明度: {alpha:.2f}")

    except Exception as e:
        print(f"\n[!] 运行出错: {e}")
    finally:
        # 7. 停止客户端，释放网络资源
        print("[*] 正在释放资源并断开连接...")
        client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
