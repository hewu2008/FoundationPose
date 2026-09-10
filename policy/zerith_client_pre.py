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
# from Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Client')
    parser.add_argument('--mesh_file', type=str,default='./model/DPPUB-204001196-AAX_01_01.obj', required=True, help='Path to mesh OBJ file')
    parser.add_argument('--server_addr', type=str, default='tcp://172.31.200.245:5555', help='ZMQ server address')
    parser.add_argument('--debug_dir', type=str, default='./client_debug', help='Directory to save debug outputs')
    parser.add_argument('--rgb_path', type=str, default="./zerith_rgb.png", help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, default="./zerith_depth.npy", help='Path to depth image')
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

    def register(self, K, rgb, depth, label=None, box=None, threshold=0.3, iteration=5):
        """Call register interface"""
        params = {
            'K': K,
            'rgb': rgb,
            'depth': depth,
            'label': label,
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


def process_label(client, K, color, depth, label, box, mesh_bbox, to_origin, label_output_dir, num_iterations=10):
    os.makedirs(f'{label_output_dir}/ob_in_cam', exist_ok=True)
    os.makedirs(f'{label_output_dir}/track_vis', exist_ok=True)
    
    print(f"\nProcessing label: {label}")
    print(f"  Box: {box}")
    print(f"  Output directory: {label_output_dir}")
    
    pose = None
    for i in range(num_iterations):
        if i == 0:
            print(f"  Registering frame {i}...")
            response = client.register(
                K=K,
                rgb=color,
                depth=depth,
                label=label,
                box=box,
                threshold=0.34,
                iteration=5
            )
            
            if response['status'] != 'success':
                print(f"  Registration failed: {response.get('message', 'Unknown error')}")
                break
            
            pose = response['pose']
            pose = pose.numpy()
            print(f"  Registration successful")
        else:
            start_time = time.time()
            response = client.track(
                K=K,
                rgb=color,
                depth=depth,
                iteration=2
            )
            
            if response['status'] != 'success':
                print(f"  Tracking failed: {response.get('message', 'Unknown error')}")
                break
            
            pose = response['pose']
            pose = pose.numpy()
            end_time = time.time()
            print(f"  Tracking frame {i} successful, time: {end_time - start_time}s")
        
        np.savetxt(f'{label_output_dir}/ob_in_cam/{i}.txt', pose.reshape(4, 4))
        
        center_pose = pose @ np.linalg.inv(to_origin)
        
        # vis = draw_posed_3d_box(K, color.copy(), center_pose, mesh_bbox)
        # vis = draw_xyz_axis(vis, center_pose, scale=0.1, K=K, thickness=3, 
        #                   transparency=0, is_input_rgb=True)
        
        # cv2.imshow('FoundationPose Tracking', vis[..., ::-1])
        # cv2.waitKey(1)
        
        # imageio.imwrite(f'{label_output_dir}/track_vis/{i}.png', vis)


def main(args):
    to_origin, mesh_bbox = load_mesh(args.mesh_file)
    
    client = create_client(args.server_addr)
    if client is None:
        return
    
    os.makedirs(args.debug_dir, exist_ok=True)
    
    try:  
        color, boxes = detect_parts(client, args.rgb_path)
        if color is None or boxes is None:
            return
        
        K_color = np.array([
            [607.62, 0.00,  329.68],
            [0.00,  608.40, 243.36],
            [0.00,  0.00, 1.00]
        ])
        
        depth = np.load(args.depth_path)
        
        for idx, box_dict in enumerate(boxes):
            label = box_dict['label']
            box = [int(box_dict['x1']), int(box_dict['y1']), int(box_dict['x2']), int(box_dict['y2'])]
            
            label_output_dir = f'{args.debug_dir}/label_{idx}'
            process_label(client, K_color, color, depth, label, box, mesh_bbox, to_origin, label_output_dir)
        
        print("\nProcessing complete")
    
    except KeyboardInterrupt:
        print("Client interrupted")
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
