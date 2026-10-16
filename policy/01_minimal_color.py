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
   checked against the calibrated state. Two points picked with the left mouse
   button on the reference panel define a line, which is mirrored onto the live
   view at the same normalized position.
* [Usage]
1. Set GRPC_TARGET to the real server IP address.
2. Set CAMERA_NAME to the corresponding camera ID on the server.
3. Set CALIB_IMAGE to the calibration image path (relative to the repo root or
   an absolute path).
* [Hotkeys]
- c : toggle image calibration mode (side-by-side view)
- left click : pick points on the reference panel (2 points draw a line on both
  panels; a 3rd click starts a new line)
- r : clear the marked line
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
LINE_COLOR = (0, 255, 0)     # color of the mouse-marked line (BGR)
LINE_THICKNESS = 2           # thickness of the mouse-marked line


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


def fit_width(image, target_h):
    """Width of `image` when scaled to `target_h`, preserving its aspect ratio."""
    return max(1, int(round(image.shape[1] * target_h / image.shape[0])))


def norm_to_pixel(point, origin_x, panel_w, panel_h):
    """Map a normalized [0, 1] coordinate to a pixel position inside a panel."""
    u, v = point
    return (int(round(origin_x + u * panel_w)), int(round(v * panel_h)))


def compose_side_by_side(frame, calib, line_points=()):
    """Left: live frame; right: reference image scaled to the same height.

    The points in `line_points` are normalized reference coordinates; they are drawn
    on both panels so the marked line can be compared between live and reference.

    Returns (canvas, ref_geometry) where ref_geometry is (origin_x, width, height)
    of the reference panel inside the canvas, used to map mouse clicks back.
    """
    h, w = frame.shape[:2]
    ref_w = fit_width(calib, h)
    ref_resized = cv2.resize(calib, (ref_w, h), interpolation=cv2.INTER_LINEAR)
    separator = np.full((h, CALIB_GAP, 3), CALIB_GAP_COLOR, dtype=np.uint8)
    canvas = np.hstack([frame[..., :3], separator, ref_resized[..., :3]])

    cv2.putText(
        canvas, "LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "REFERENCE", (w + CALIB_GAP + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )

    # Draw the marked points/line on both panels at the same normalized position
    for origin_x, panel_w in ((0, w), (w + CALIB_GAP, ref_w)):
        pixels = [norm_to_pixel(p, origin_x, panel_w, h) for p in line_points]
        for pixel in pixels:
            cv2.circle(canvas, pixel, 6, LINE_COLOR, 2, cv2.LINE_AA)
        if len(pixels) == 2:
            cv2.line(canvas, pixels[0], pixels[1], LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    return canvas, (w + CALIB_GAP, ref_w, h)


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

    # --- Mouse-marked line: points are stored as normalized [0, 1] reference coordinates ---
    line_points = []
    ref_geom = None                  # (origin_x, width, height) of the reference panel on screen

    def on_mouse(event, x, y, flags, param):
        nonlocal ref_geom
        if event != cv2.EVENT_LBUTTONDOWN or ref_geom is None:
            return                   # clicks are only meaningful in side-by-side mode
        origin_x, panel_w, panel_h = ref_geom
        if not (origin_x <= x < origin_x + panel_w and 0 <= y < panel_h):
            return                   # only clicks on the reference panel are accepted
        if len(line_points) >= 2:
            line_points.clear()      # a third click starts a new line
        line_points.append(((x - origin_x) / panel_w, y / panel_h))
        print(f"[*] Point {len(line_points)} on reference: "
              f"(u={line_points[-1][0]:.3f}, v={line_points[-1][1]:.3f})")

    # Create the window up front so the mouse callback can be attached before the first frame
    cv2.namedWindow("Color Viewer", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Color Viewer", on_mouse)

    try:
        # 3. Start the client (gRPC handshake and WebRTC setup)
        print(f"[*] Connecting to server {GRPC_TARGET} ...")
        client.start()
        print(f"[+] Connected. Streaming from camera [{CAMERA_NAME}] ...")
        print("[i] Hotkeys: c toggle calibration view / click reference to mark a line / r clear / q quit")

        while True:
            # 4. Fetch the latest color frame (returns (numpy_bgr_array, timestamp) or None)
            data = client.get_latest_frame(CAMERA_NAME)

            if data:
                img, ts = data
                display = img

                # 5. Image calibration mode: show the reference image beside the live frame
                geom = None
                if calib_mode and calib_image is not None:
                    display, geom = compose_side_by_side(img, calib_image, line_points)
                ref_geom = geom

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
            elif key == ord('r'):
                line_points.clear()
                print("[*] Cleared the marked line.")

    except Exception as e:
        print(f"\n[!] Runtime error: {e}")
    finally:
        # 7. Stop the client and release network resources
        print("[*] Releasing resources and disconnecting...")
        client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
