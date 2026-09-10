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
#from Utils import *
#from zerith.zerith_mesh_utils import load_trimesh

def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Client')
    parser.add_argument(
        '--mesh_file', type=str,default='./model/DPPUB-204001196-AAX_01_01.obj', help=argparse.SUPPRESS
    )  # Backward-compatible; each detection now supplies its own mesh.
    parser.add_argument('--server_addr', type=str, default='tcp://192.168.3.28:5555', help='ZMQ server address')
    parser.add_argument('--debug_dir', type=str, default='./client_debug', help='Directory to save debug outputs')
    parser.add_argument('--rgb_path', type=str, default="./zerith_rgb.png", help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, default="./zerith_depth.npy", help='Path to depth image')
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
    mesh = trimesh.load(mesh_file)
    #mesh = load_trimesh(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    mesh_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    print(f"Loaded mesh: {mesh_file}")
    print(f"Mesh extents: {extents}")
    print(f"Mesh bbox: {mesh_bbox}")
    return to_origin, mesh_bbox


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
    if response['status'] != 'success':
        print(f"  Registration failed: {response.get('message', 'Unknown error')}")
        return False

    pose = response['pose']
    pose = pose.numpy() if hasattr(pose, 'numpy') else np.asarray(pose)
    angle = response['blue_z_plane_angle_deg']
    print(f"  Registration successful, time: {elapsed:.3f}s")
    print(f"  SAM tight box: {response.get('mask_box')}")

    filepath = f'{label_output_dir}/ob_in_cam/0.txt'
    np.savetxt(filepath, pose.reshape(4, 4))
    with open(filepath, 'a') as f:
        f.write(f'{angle}\n')
    #center_pose = pose @ np.linalg.inv(to_origin)
    # vis = draw_posed_3d_box(K, color.copy(), center_pose, mesh_bbox)
    # vis = draw_xyz_axis(
    #     vis,
    #     center_pose,
    #     scale=0.1,
    #     K=K,
    #     thickness=3,
    #     transparency=0,
    #     is_input_rgb=True,
    # )
    # if show:
    #     cv2.imshow('FoundationPose Registration', vis[..., ::-1])
    #     cv2.waitKey(1)
    # imageio.imwrite(f'{label_output_dir}/track_vis/0.png', vis)
    return True


def save_detection_debug(debug_dir, color, boxes):
    """Save a visualization of the server detections used by the client."""
    vis = color.copy()
    colors = {
        'cat2': (0, 255, 0),
        'cat3': (0, 128, 255),
        'cat4': (255, 255, 0),
    }
    for box in boxes:
        x1, y1, x2, y2 = [
            int(round(float(box[key]))) for key in ('x1', 'y1', 'x2', 'y2')
        ]
        category_id = str(box.get('category_id', 'part'))
        color_value = colors.get(category_id, (255, 0, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_value, 2)
        cv2.putText(
            vis,
            category_id,
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
        for box_dict in boxes:
            label = box_dict['label']
            category_id = box_dict['category_id']
            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1
            mesh_file = box_dict['mesh_file']
            to_origin, mesh_bbox = None, None

            box = [int(box_dict['x1']), int(box_dict['y1']), int(box_dict['x2']), int(box_dict['y2'])]
            
            label_output_dir = (
                f'{args.debug_dir}/{category_id}_{instance_index}'
            )
            process_label(
                client, K_color, color, depth, label, category_id,
                box, mesh_bbox, to_origin, label_output_dir,
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
