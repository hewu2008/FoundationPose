"""
@file 01_minimal_color.py
@brief [Demo 01] Minimal color frame capture + image calibration overlay
* [Overview]
1. Shows how to initialize a CameraClient and establish a connection.
2. Demonstrates non-blocking retrieval of the latest color image (RGB) and
   displaying it with OpenCV.
3. Adds an "image calibration mode": shows a calibration image loaded from a
   fixed path side by side with the live camera frame (live on the left,
   reference on the right, heights matched), so the camera can be visually
   checked against the calibrated state.
* [Usage]
1. Set GRPC_TARGET to the real server IP address.
2. Set CAMERA_NAME to the corresponding camera ID on the server.
3. Set CALIB_IMAGE to the calibration image path (relative to the repo root or
   an absolute path).
* [Hotkeys]
- c : toggle image calibration mode (side-by-side view)
- q : quit the preview
"""

import os
import sys

import cv2
import numpy as np

# --- Path compatibility: make the parent directory containing camera_client importable ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
from camera_client import CameraClient

# --- Image calibration mode configuration ---
# Calibration image path: a relative path resolves against the repo root, an absolute path is used as-is.
CALIB_IMAGE = "assets/imgs/zerith_ref_rgb.png"
CALIB_GAP = 8                # width in pixels of the separator between the two panels
CALIB_GAP_COLOR = 32         # separator color (grayscale value)


def resolve_path(path):
    """Resolve a relative path against the repo root; absolute paths are returned as-is."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_calibration_image(path):
    """Load a calibration image as a 3-channel BGR array; return None on failure."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:                       # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                 # BGRA -> composite onto black using its alpha
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    else:
        img = np.ascontiguousarray(img[..., :3])
    return img


def compose_side_by_side(frame, calib):
    """Place the live frame on the left and the calibration image on the right, matching heights."""
    h, w = frame.shape[:2]
    calib_h, calib_w = calib.shape[:2]
    new_w = max(1, int(round(calib_w * h / calib_h)))
    calib_resized = cv2.resize(calib, (new_w, h), interpolation=cv2.INTER_LINEAR)
    separator = np.full((h, CALIB_GAP, 3), CALIB_GAP_COLOR, dtype=np.uint8)
    canvas = np.hstack([frame[..., :3], separator, calib_resized[..., :3]])
    cv2.putText(
        canvas, "LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "REFERENCE", (w + CALIB_GAP + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def main():
    # ---------------------------------------------------------
    # [User config] adjust these for your environment
    # ---------------------------------------------------------
    GRPC_TARGET = "localhost:50051"  # server IP:port
    CAMERA_NAME = "rs/cam_high"      # color camera ID
    # ---------------------------------------------------------

    # 1. Initialize the client
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

    # 2. Preload the calibration image (fixed path); calibration mode is disabled if it fails
    calib_path = resolve_path(CALIB_IMAGE)
    calib_image = load_calibration_image(calib_path)
    if calib_image is None:
        print(f"[!] Failed to load calibration image [{calib_path}]; 'c' will not enable calibration mode.")
    else:
        print(f"[*] Loaded calibration image: {calib_path}")
    calib_mode = False               # image calibration mode toggle

    try:
        # 3. Start the client (gRPC handshake and WebRTC setup)
        print(f"[*] Connecting to server {GRPC_TARGET} ...")
        client.start()
        print(f"[+] Connected. Streaming from camera [{CAMERA_NAME}] ...")
        print("[i] Hotkeys: c toggle calibration view / q quit")

        while True:
            # 4. Fetch the latest color frame (returns (numpy_bgr_array, timestamp) or None)
            data = client.get_latest_frame(CAMERA_NAME)

            if data:
                img, ts = data
                display = img

                # 5. Image calibration mode: show the reference image beside the live frame
                if calib_mode and calib_image is not None:
                    display = compose_side_by_side(img, calib_image)

                show_image("Color Viewer", display)

            # 6. Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[+] User requested exit.")
                break
            elif key == ord('c'):
                if calib_image is None:
                    print(f"[!] Calibration image unavailable; cannot enable calibration mode: {calib_path}")
                else:
                    calib_mode = not calib_mode
                    print(f"[*] Image calibration mode: {'ON' if calib_mode else 'OFF'}")

    except Exception as e:
        print(f"\n[!] Runtime error: {e}")
    finally:
        # 7. Stop the client and release network resources
        print("[*] Releasing resources and disconnecting...")
        client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
