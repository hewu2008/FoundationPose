# encoding:utf8
import argparse
import zmq
import pickle
import numpy as np
import cv2
import os
import imageio
import trimesh
from datareader import YcbineoatReader
from Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Client')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video scene directory')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to mesh OBJ file')
    parser.add_argument('--server_addr', type=str, default='tcp://localhost:5555', help='ZMQ server address')
    parser.add_argument('--debug_dir', type=str, default='./client_debug', help='Directory to save debug outputs')
    parser.add_argument('--labels', type=str, nargs='+', default=["object."], help='Detection labels for registration')
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
    
    def register(self, K, rgb, depth, labels=None, threshold=0.3, iteration=5):
        """Call register interface"""
        params = {
            'K': K,
            'rgb': rgb,
            'depth': depth,
            'labels': labels if labels else ["object."],
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
    
    def ping(self):
        """Check if server is alive"""
        return self.send_request('ping')
    
    def close(self):
        """Close connection"""
        self.socket.close()
        self.context.term()


def main(args):
    # Load mesh and compute bounding box
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    print(f"Loaded mesh: {args.mesh_file}")
    print(f"Mesh extents: {extents}")
    print(f"Mesh bbox: {bbox}")
    
    # Create client
    client = ZerithFoundationPoseClient(args.server_addr)
    
    # Check server connection
    response = client.ping()
    if response['status'] != 'success':
        print(f"Server connection failed: {response.get('message', 'Unknown error')}")
        return
    
    print("Server connection established successfully")
    
    # Initialize reader
    reader = YcbineoatReader(video_dir=args.video_dir, shorter_side=None, zfar=np.inf)
    
    # Create debug directories
    os.makedirs(args.debug_dir, exist_ok=True)
    os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
    os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
    
    try:
        for i in range(len(reader.color_files)):
            print(f"Processing frame {i}/{len(reader.color_files)}")
            
            # Get frame data
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            
            if i == 0:
                # First frame: register
                print(f"Registering frame {i}...")
                response = client.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    labels=args.labels,
                    threshold=0.3,
                    iteration=5
                )
                
                if response['status'] != 'success':
                    print(f"Registration failed: {response.get('message', 'Unknown error')}")
                    break
                
                pose = response['pose']
                print(f"Registration successful")
            else:
                # Subsequent frames: track
                start_time = time.time()
                response = client.track(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    iteration=2
                )
                
                if response['status'] != 'success':
                    print(f"Tracking failed: {response.get('message', 'Unknown error')}")
                    break
                
                pose = response['pose']
                end_time = time.time()
                print(f"Tracking frame {i} successful, time: {end_time - start_time}s")
            
            # Save pose
            np.savetxt(f'{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))
            
            # Compute center pose for visualization
            center_pose = pose @ np.linalg.inv(to_origin)
            
            # Draw 3D bounding box and XYZ axis
            vis = draw_posed_3d_box(reader.K, color.copy(), center_pose, bbox)
            vis = draw_xyz_axis(vis, center_pose, scale=0.1, K=reader.K, thickness=3, 
                              transparency=0, is_input_rgb=True)
            
            # Display visualization
            cv2.imshow('FoundationPose Tracking', vis[..., ::-1])
            cv2.waitKey(1)
            
            # Save visualization
            imageio.imwrite(f'{args.debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
        
        print("Processing complete")
    
    except KeyboardInterrupt:
        print("Client interrupted")
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
