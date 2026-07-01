# encoding:utf8
import zmq
import pickle
import logging
from typing import Dict, Any
import torch
import cv2
import os
import numpy as np


class ZerithServer:
    def __init__(self, pose_estimator, segmentation, save_dir='./debug_register'):
        self.pose_estimator = pose_estimator
        self.segmentation = segmentation
        self.is_initialized = False
        self._call_index = 0
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_overlay'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ob_in_cam'), exist_ok=True)

    def _save_debug_images(self, index, rgb, ob_mask):
        try:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_dir, 'rgb', f'{index}.png'), rgb_bgr)

            mask_vis = rgb.copy()
            mask_vis[ob_mask] = [0, 255, 0]
            mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_dir, 'mask_overlay', f'{index}.png'), mask_vis_bgr)

            logging.info(f"Debug images saved for index {index}")
        except Exception as save_e:
            logging.warning(f"Failed to save debug images: {str(save_e)}")

    def _handle_register(self, request):
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            labels = request.get('labels', ["object."])
            threshold = request.get('threshold', 0.3)
            iteration = request.get('iteration', 5)

            logging.info("Received register command with automatic segmentation")

            ob_mask = self.segmentation.segment(rgb, labels, threshold)

            if ob_mask is None:
                logging.warning("Automatic segmentation failed, using depth-based mask")
                ob_mask = (depth > 0).astype(bool)
            else:
                if isinstance(ob_mask, bytes):
                    ob_mask = np.frombuffer(ob_mask, dtype=np.uint8).reshape(rgb.shape[:2])
                ob_mask = (ob_mask > 0).astype(bool)

            index = self._call_index
            self._call_index += 1

            self._save_debug_images(index, rgb, ob_mask)

            pose = self.pose_estimator.register(K, rgb, depth, ob_mask, iteration)
            np.savetxt(f'{self.save_dir}/ob_in_cam/{index}.txt', pose.reshape(4, 4))
            pose = torch.from_numpy(pose) if isinstance(pose, type(None)) == False else pose

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
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            iteration = request.get('iteration', 2)

            logging.info("Received track command")

            pose = self.pose_estimator.track(K, rgb, depth, iteration)
            pose = torch.from_numpy(pose) if isinstance(pose, type(None)) == False else pose

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
        logging.debug("Received ping command")
        return {
            'status': 'success',
            'message': 'Server is running'
        }

    def _handle_unknown(self, command):
        logging.warning(f"Unknown command: {command}")
        return {
            'status': 'error',
            'message': f'Unknown command: {command}'
        }

    def _process_request(self, message):
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

    def start(self, port=5555):
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
