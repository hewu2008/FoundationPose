"""
@file 01_minimal_color.py
@brief [Demo 01] Minimal color frame capture + image calibration overlay
* [Overview]
1. Shows how to initialize a CameraClient and establish a connection.
2. Demonstrates non-blocking retrieval of the latest color image (RGB) and
   displaying it with OpenCV.
3. Adds an "image calibration mode": overlays a calibration image loaded from a
   fixed path semi-transparently onto the live camera frame, so the camera can
   be visually checked against the calibrated state. The overlay opacity is
   adjustable at runtime.
* [Usage]
1. Set GRPC_TARGET to the real server IP address.
2. Set CAMERA_NAME to the corresponding camera ID on the server.
3. Set CALIB_IMAGE to the calibration image path (relative to the repo root or
   an absolute path).
* [Hotkeys]
- c : toggle image calibration mode (semi-transparent overlay)
- [ : decrease calibration overlay opacity
- ] : increase calibration overlay opacity
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
CALIB_IMAGE = "assets/imgs/zero.png"
CALIB_ALPHA_DEFAULT = 0.5   # initial calibration overlay opacity, range [0, 1]
CALIB_ALPHA_STEP = 0.05     # opacity step applied per key press


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


def blend_calibration(frame, calib, alpha):
    """Blend the calibration image over the camera frame with the given opacity."""
    return cv2.addWeighted(frame, 1.0 - alpha, calib, alpha, 0.0)


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
    calib_fitted = {}                # cache of the calibration image resized per frame size
    calib_mode = False               # image calibration mode toggle
    alpha = CALIB_ALPHA_DEFAULT      # calibration overlay opacity

    try:
        # 3. Start the client (gRPC handshake and WebRTC setup)
        print(f"[*] Connecting to server {GRPC_TARGET} ...")
        client.start()
        print(f"[+] Connected. Streaming from camera [{CAMERA_NAME}] ...")
        print("[i] Hotkeys: c toggle calibration overlay / [ ] opacity / q quit")

        while True:
            # 4. Fetch the latest color frame (returns (numpy_bgr_array, timestamp) or None)
            data = client.get_latest_frame(CAMERA_NAME)

            if data:
                img, ts = data
                display = img

                # 5. Image calibration mode: overlay the calibration image semi-transparently
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
            elif key == ord('['):
                alpha = max(0.0, alpha - CALIB_ALPHA_STEP)
                print(f"[*] Calibration overlay opacity: {alpha:.2f}")
            elif key == ord(']'):
                alpha = min(1.0, alpha + CALIB_ALPHA_STEP)
                print(f"[*] Calibration overlay opacity: {alpha:.2f}")

    except Exception as e:
        print(f"\n[!] Runtime error: {e}")
    finally:
        # 7. Stop the client and release network resources
        print("[*] Releasing resources and disconnecting...")
        client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
