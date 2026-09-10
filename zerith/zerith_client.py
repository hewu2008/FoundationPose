# encoding:utf8
import argparse
import zmq
import pickle
import numpy as np
import cv2
import os
import imageio
import trimesh
import time
from Utils import *
from zerith.zerith_mesh_utils import load_trimesh, blue_z_plane_angle_deg

def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Client')
    parser.add_argument(
        '--mesh_file', default=None, help=argparse.SUPPRESS
    )  # Backward-compatible; each detection now supplies its own mesh.
    parser.add_argument('--server_addr', type=str, default='tcp://localhost:5555', help='ZMQ server address')
    parser.add_argument('--debug_dir', type=str, default='./client_debug', help='Directory to save debug outputs')
    parser.add_argument('--rgb_path', type=str, default="/home/chery/gzl/jszn/Foundation_file/FoundationPose/assets/zerith_rgb.png", help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, default="/home/chery/gzl/jszn/Foundation_file/FoundationPose/assets/zerith_depth.npy", help='Path to depth image')
    parser.add_argument('--register_iterations', type=int, default=5, help='FoundationPose refinement iterations for the single offline registration')
    parser.add_argument('--show', action='store_true', help='Show the registration window (disabled by default for headless runs)')
    return parser.parse_args()

class ZerithFoundationPoseClient:
    def __init__(self, server_addr):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_addr)
        print(f"Connected to server: {server_addr}")
    
    def send_request(self, command, **kwargs):
        """Send request to server and receive response"""
        request = {'command': command}
        request.update(kwargs)
        
        try:
            self.socket.send(pickle.dumps(request))
            response = self.socket.recv()
            return pickle.loads(response)
        except Exception as e:
            print(f"Error sending request: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def ping(self):
        """Check if server is alive"""
        return self.send_request('ping')
    
    def detection(self, rgb):
        """Call detection interface for part detection"""
        params = {
            'rgb': rgb
        }
        return self.send_request('detection', **params)

    def register(self, K, rgb, depth, label=None, category_id=None, box=None, threshold=0.3, iteration=5):
        """Call register interface"""
        params = {
            'K': K,
            'rgb': rgb,
            'depth': depth,
            'label': label,
            'category_id': category_id,
            'box': box,
            'threshold': threshold,
            'iteration': iteration
        }
        return self.send_request('register', **params)

    def register_batch(
        self,
        K,
        rgb,
        depth,
        objects,
        debug_image_policy=None,
    ):
        """Upload one RGB-D frame and register all supplied objects."""
        params = {
            'K': K,
            'rgb': rgb,
            'depth': depth,
            'objects': objects,
        }
        if debug_image_policy is not None:
            params['debug_image_policy'] = debug_image_policy
        return self.send_request('register_batch', **params)
    
    def track(self, K, rgb, depth, iteration=2):
        """Call track interface"""
        params = {
            'K': K,
            'rgb': rgb,
            'depth': depth,
            'iteration': iteration
        }
        return self.send_request('track', **params)
    
    def close(self):
        """Close connection"""
        self.socket.close()
        self.context.term()

def load_mesh(mesh_file):
    mesh = load_trimesh(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    mesh_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    print(f"Loaded mesh: {mesh_file}")
    print(f"Mesh extents: {extents}")
    print(f"Mesh bbox: {mesh_bbox}")
    return to_origin, mesh_bbox


def build_instance_output_dir(debug_dir, category_id, instance_index=None):
    """Return ``category_id_instance_index`` for an offline result."""
    if instance_index is None:
        # Preserve the older two-argument helper call used by copied clients.
        return os.path.join(str(debug_dir), str(category_id))
    return os.path.join(
        str(debug_dir),
        f'{category_id}_{int(instance_index)}',
    )


def create_client(server_addr):
    client = ZerithFoundationPoseClient(server_addr)
    response = client.ping()
    if response['status'] != 'success':
        print(f"Server connection failed: {response.get('message', 'Unknown error')}")
        return None
    print("Server connection established successfully")
    return client


def detect_parts(client, rgb_path):
    color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    response = client.detection(color)
    if response['status'] != 'success':
        print(f"Detection failed: {response.get('message', 'Unknown error')}")
        return None, None
    
    print(f"Detection successful, found {len(response['boxes'])} boxes")
    return color, response['boxes']


def process_label(
    client,
    K,
    color,
    depth,
    label,
    category_id,
    box,
    mesh_bbox,
    to_origin,
    label_output_dir,
    register_iterations=5,
    show=False,
):
    os.makedirs(f'{label_output_dir}/ob_in_cam', exist_ok=True)
    os.makedirs(f'{label_output_dir}/track_vis', exist_ok=True)
    
    print(f"\nProcessing label: {label}")
    print(f"  Box: {box}")
    print(f"  Output directory: {label_output_dir}")
    
    print("  Registering offline frame once...")
    start_time = time.time()
    response = client.register(
        K=K,
        rgb=color,
        depth=depth,
        label=label,
        category_id=category_id,
        box=box,
        threshold=0.34,
        iteration=register_iterations,
    )
    elapsed = time.time() - start_time
    return save_registration_result(
        K=K,
        color=color,
        response=response,
        mesh_bbox=mesh_bbox,
        to_origin=to_origin,
        label_output_dir=label_output_dir,
        category_id=category_id,
        show=show,
        elapsed=elapsed,
    )


def save_registration_result(
    K,
    color,
    response,
    mesh_bbox,
    to_origin,
    label_output_dir,
    category_id=None,
    show=False,
    elapsed=None,
):
    """Persist and visualize one result from single or batch registration."""
    os.makedirs(f'{label_output_dir}/ob_in_cam', exist_ok=True)
    os.makedirs(f'{label_output_dir}/track_vis', exist_ok=True)

    if response['status'] != 'success':
        print(f"  Registration failed: {response.get('message', 'Unknown error')}")
        return False

    pose = response['pose']
    pose = pose.numpy() if hasattr(pose, 'numpy') else np.asarray(pose)
    if elapsed is None:
        elapsed = response.get('elapsed_seconds')
    if elapsed is None:
        print("  Registration successful")
    else:
        print(f"  Registration successful, time: {float(elapsed):.3f}s")
    print(f"  Instance-mask tight box: {response.get('mask_box')}")

    np.savetxt(f'{label_output_dir}/ob_in_cam/0.txt', pose.reshape(4, 4))

    # ``1.txt`` now stores only the projected blue +Z-axis plane angle.  The
    # server supplies it in the response; the local fallback keeps this
    # client compatible with an older server during a rolling update.
    plane_angle = response.get('blue_z_plane_angle_deg')
    if plane_angle is None:
        try:
            axis_to_origin = None if str(category_id) == 'cat5' else to_origin
            plane_angle = blue_z_plane_angle_deg(
                K,
                pose,
                to_origin=axis_to_origin,
            )
        except ValueError as angle_error:
            print(f"  Blue +Z plane angle unavailable: {angle_error}")
    if plane_angle is not None:
        plane_angle = float(plane_angle)
        np.savetxt(
            f'{label_output_dir}/ob_in_cam/1.txt',
            np.array([plane_angle], dtype=np.float64),
            fmt='%.8f',
        )
        print(
            "  Blue +Z plane angle: "
            f"{plane_angle:.6f} deg (90 deg reference, clockwise positive, "
            "folded to [-90, 90])"
        )
    else:
        # Do not leave a previous center-pose matrix in the new scalar-angle
        # slot when the projected direction is mathematically undefined.
        np.savetxt(
            f'{label_output_dir}/ob_in_cam/1.txt',
            np.array([np.nan], dtype=np.float64),
            fmt='%.8f',
        )

    if to_origin is None or mesh_bbox is None:
        print("  Visualization skipped: mesh bounds/origin were not supplied")
        return True

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, color.copy(), center_pose, mesh_bbox)
    # The cat5 elbow pipe uses its authored CAD frame for downstream grasp
    # semantics.  Keep the enclosing cuboid in the compact OBB frame, but do
    # not rotate or recenter the displayed XYZ axes away from the CAD frame.
    axis_pose = pose if str(category_id) == 'cat5' else center_pose
    vis = draw_xyz_axis(
        vis,
        axis_pose,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    if show:
        cv2.imshow('FoundationPose Registration', vis[..., ::-1])
        cv2.waitKey(1)
    imageio.imwrite(f'{label_output_dir}/track_vis/0.png', vis)
    return True


def process_labels_batch(
    client,
    K,
    color,
    depth,
    items,
    register_iterations=5,
    show=False,
):
    """Register all prepared instances with one RGB-D upload.

    Each item must contain label/category_id/box, mesh_bbox/to_origin and
    label_output_dir.  The returned booleans preserve the request order.
    """
    if not items:
        return []

    objects = []
    for item in items:
        print(f"\nProcessing label: {item['label']}")
        print(f"  Category: {item['category_id']}")
        print(f"  Box: {item['box']}")
        print(f"  Output directory: {item['label_output_dir']}")
        request_object = {
            'label': item['label'],
            'category_id': item['category_id'],
            'box': item['box'],
            'iteration': int(item.get(
                'register_iterations', register_iterations
            )),
        }
        for key in ('detection_id', 'mask_id'):
            if item.get(key) is not None:
                request_object[key] = item[key]
        objects.append(request_object)

    print(
        f"\nBatch registering {len(objects)} objects with one RGB-D upload..."
    )
    started = time.time()
    response = client.register_batch(K, color, depth, objects)
    elapsed = time.time() - started
    if response.get('status') != 'success':
        print(
            "Batch registration failed: "
            f"{response.get('message', 'Unknown error')}"
        )
        return [False] * len(items)

    results = response.get('results', [])
    if len(results) != len(items):
        print(
            f"Batch registration returned {len(results)} results for "
            f"{len(items)} objects"
        )
        return [False] * len(items)

    timing = response.get('timing', {})
    print(
        "Batch registration completed, "
        f"round-trip={elapsed:.3f}s, "
        f"Mask lookup={float(timing.get('segmentation_seconds', 0.0)):.3f}s, "
        f"FoundationPose={float(timing.get('registration_seconds', 0.0)):.3f}s"
    )

    successes = []
    for item, result in zip(items, results):
        print(
            f"\nSaving {item['category_id']} result "
            f"(server debug index={result.get('debug_index')})..."
        )
        successes.append(save_registration_result(
            K=K,
            color=color,
            response=result,
            mesh_bbox=item.get('mesh_bbox'),
            to_origin=item.get('to_origin'),
            label_output_dir=item['label_output_dir'],
            category_id=item.get('category_id'),
            show=show,
            elapsed=result.get('elapsed_seconds'),
        ))
    return successes


def save_detection_debug(debug_dir, color, boxes):
    """Save a visualization of the server detections used by the client."""
    vis = color.copy()
    colors = {
        'cat2': (0, 255, 0),
        'cat3': (0, 128, 255),
        'cat4': (255, 255, 0),
    }
    category_counts = {}
    for box in boxes:
        x1, y1, x2, y2 = [
            int(round(float(box[key]))) for key in ('x1', 'y1', 'x2', 'y2')
        ]
        category_id = str(box.get('category_id', 'part'))
        instance_index = category_counts.get(category_id, 0)
        category_counts[category_id] = instance_index + 1
        color_value = colors.get(category_id, (255, 0, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_value, 2)
        cv2.putText(
            vis,
            f'{category_id}_{instance_index}',
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_value,
            1,
            cv2.LINE_AA,
        )
    imageio.imwrite(os.path.join(debug_dir, 'detections.png'), vis)


def main(args):
    client = create_client(args.server_addr)
    if client is None:
        return
    
    os.makedirs(args.debug_dir, exist_ok=True)
    
    try:  
        color, boxes = detect_parts(client, args.rgb_path)
        if color is None or boxes is None:
            return
        save_detection_debug(args.debug_dir, color, boxes)
        
        K_color = np.array([
            [607.62, 0.00,  329.68],
            [0.00,  608.40, 243.36],
            [0.00,  0.00, 1.00]
        ])
        
        depth = np.load(args.depth_path)
        print(f"depth dtype: {depth.dtype}, K_color dtype: {K_color.dtype}")
        
        category_counts = {}
        batch_items = []
        for box_dict in boxes:
            label = box_dict['label']
            category_id = str(box_dict['category_id'])
            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1
            mesh_file = box_dict['mesh_file']
            to_origin, mesh_bbox = load_mesh(mesh_file)

            # # TODO: debug only, delete later
            # print(f"category_id = {category_id}, instance_index = {instance_index}")
            # if not (str(category_id) == "cat5" and int(instance_index) == 0):
            #     continue

            box = [int(box_dict['x1']), int(box_dict['y1']), int(box_dict['x2']), int(box_dict['y2'])]
            
            label_output_dir = build_instance_output_dir(
                args.debug_dir,
                category_id,
                instance_index,
            )
            batch_items.append({
                'label': label,
                'category_id': category_id,
                'box': box,
                'detection_id': box_dict.get('detection_id'),
                'mask_id': box_dict.get('mask_id'),
                'mesh_bbox': mesh_bbox,
                'to_origin': to_origin,
                'label_output_dir': label_output_dir,
            })

        process_labels_batch(
            client=client,
            K=K_color,
            color=color,
            depth=depth,
            items=batch_items,
            register_iterations=args.register_iterations,
            show=args.show,
        )
        
        print("\nProcessing complete")
    
    except KeyboardInterrupt:
        print("Client interrupted")
    finally:
        client.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
