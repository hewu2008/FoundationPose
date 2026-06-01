# encoding:utf8
import argparse
import trimesh
import logging
import numpy as np
import zmq
import pickle
import cv2
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
from estimater import FoundationPose


def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Server')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the mesh file (OBJ format)')
    parser.add_argument('--zmq_port', type=int, default=5555, help='ZMQ server port')
    return parser.parse_args()


class ZerithFoundationPoseServer:
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        logging.info(f"Mesh loaded: {self.mesh_file}")

        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2,3)
        logging.info(f"Mesh bbox: {self.bbox}")

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        logging.info(f"Load scorer, refiner and glctx successfully")

        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices, 
            model_normals=self.mesh.vertex_normals, 
            mesh=self.mesh, 
            scorer=self.scorer, 
            refiner=self.refiner, 
            debug_dir=None, 
            debug=0, 
            glctx=self.glctx
        )
        logging.info(f"Load estimator successfully")
        
        self.is_initialized = False

    def register(self, K, rgb, depth, ob_mask, iteration=5):
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=iteration)
        self.is_initialized = True
        return pose
    
    def track(self, K, rgb, depth, iteration=2):
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call register first.")
        return self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
    
    def start(self, port=5555):
        """Start ZMQ server with register and track interfaces"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        logging.info(f"ZMQ server started on port {port}")
        
        try:
            while True:
                message = socket.recv()
                response = self._process_request(message)
                socket.send(pickle.dumps(response))
        except KeyboardInterrupt:
            logging.info("Server shutting down")
        finally:
            socket.close()
            context.term()
    
    def _handle_register(self, request):
        """Handle register command"""
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            ob_mask = request.get('ob_mask', None)
            iteration = request.get('iteration', 5)
            
            logging.info("Received register command")
            
            if ob_mask is None:
                ob_mask = (depth > 0).astype(bool)
            
            pose = self.register(K, rgb, depth, ob_mask, iteration)
            
            return {
                'status': 'success',
                'pose': pose,
                'message': 'Registration successful'
            }
        except Exception as e:
            logging.error(f"Error in register: {str(e)}")
            return {
                'status': 'error',
                'message': f'Registration failed: {str(e)}'
            }
    
    def _handle_track(self, request):
        """Handle track command"""
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            iteration = request.get('iteration', 2)
            
            logging.info("Received track command")
            
            pose = self.track(K, rgb, depth, iteration)
            
            return {
                'status': 'success',
                'pose': pose,
                'message': 'Tracking successful'
            }
        except Exception as e:
            logging.error(f"Error in track: {str(e)}")
            return {
                'status': 'error',
                'message': f'Tracking failed: {str(e)}'
            }
    
    def _handle_ping(self, request):
        """Handle ping command"""
        logging.debug("Received ping command")
        return {
            'status': 'success',
            'message': 'Server is running'
        }
    
    def _handle_unknown(self, command):
        """Handle unknown command"""
        logging.warning(f"Unknown command: {command}")
        return {
            'status': 'error',
            'message': f'Unknown command: {command}'
        }
    
    def _process_request(self, message):
        """Process incoming request and dispatch to appropriate handler"""
        try:
            request = pickle.loads(message)
            command = request.get('command', '')
            
            handlers = {
                'register': self._handle_register,
                'track': self._handle_track,
                'ping': self._handle_ping
            }
            
            handler = handlers.get(command, lambda req: self._handle_unknown(command))
            return handler(request)
            
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    


def main(args): 
    server = ZerithFoundationPoseServer(mesh_file=args.mesh_file)
    server.start(port=args.zmq_port)


if __name__ == '__main__':
    args = parse_args()
    main(args)
