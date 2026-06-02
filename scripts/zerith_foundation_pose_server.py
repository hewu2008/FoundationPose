# encoding:utf8
import argparse
import trimesh
import logging
import numpy as np
import zmq
import pickle
import cv2
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
from estimater import FoundationPose


def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Server')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the mesh file (OBJ format)')
    parser.add_argument('--zmq_port', type=int, default=5555, help='ZMQ server port')
    parser.add_argument('--detector_id', type=str, default="IDEA-Research/grounding-dino-tiny", help='Grounding DINO model ID')
    parser.add_argument('--segmenter_id', type=str, default="facebook/sam-vit-base", help='Segment Anything model ID')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    return parser.parse_args()


class DetectionResult:
    def __init__(self, score: float, label: str, box: List[int], mask: Optional[np.array] = None):
        self.score = score
        self.label = label
        self.box = box  # [xmin, ymin, xmax, ymax]
        self.mask = mask


class ZerithFoundationPoseServer:
    def __init__(self, mesh_file, detector_id=None, segmenter_id=None):
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
            debug_dir=args.debug_dir, 
            debug=0, 
            glctx=self.glctx
        )
        logging.info(f"Load estimator successfully")
        
        self.is_initialized = False
        
        # Initialize Grounding DINO and SAM for automatic segmentation
        self._init_segmentation_models(detector_id, segmenter_id)

    def _init_segmentation_models(self, detector_id=None, segmenter_id=None):
        """Initialize Grounding DINO detector and SAM segmenter"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
        self.segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
        
        logging.info(f"Loading Grounding DINO: {self.detector_id}")
        self.object_detector = pipeline(
            model=self.detector_id, 
            task="zero-shot-object-detection", 
            device=self.device
        )
        
        logging.info(f"Loading SAM: {self.segmenter_id}")
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)
        
        logging.info("Segmentation models loaded successfully")

    def _refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)
        
        if polygon_refinement:
            for idx, mask in enumerate(masks):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = largest_contour.reshape(-1, 2).tolist()
                    new_mask = np.zeros(mask.shape, dtype=np.uint8)
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(new_mask, [pts], color=255)
                    masks[idx] = new_mask
        
        return masks

    def _detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
        """Use Grounding DINO to detect objects"""
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.object_detector(image, candidate_labels=labels, threshold=threshold)
        
        detections = []
        for result in results:
            box = [
                result['box']['xmin'],
                result['box']['ymin'],
                result['box']['xmax'],
                result['box']['ymax']
            ]
            detections.append(DetectionResult(
                score=result['score'],
                label=result['label'],
                box=box
            ))
        
        return detections

    def _segment(self, image: Image.Image, detections: List[DetectionResult], polygon_refinement: bool = False) -> List[DetectionResult]:
        """Use SAM to generate masks"""
        if not detections:
            return detections
        
        boxes = [[det.box] for det in detections]
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)
        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        
        masks = self._refine_masks(masks, polygon_refinement)
        
        for detection, mask in zip(detections, masks):
            detection.mask = mask
        
        return detections

    def _auto_segment(self, rgb: np.ndarray, labels: List[str], threshold: float = 0.3) -> Optional[np.ndarray]:
        """Perform automatic segmentation using Grounding DINO + SAM"""
        # Convert numpy array to PIL Image
        image = Image.fromarray(rgb.astype(np.uint8))
        
        # Detect objects
        detections = self._detect(image, labels, threshold)
        
        if not detections:
            logging.warning("No objects detected")
            return None
        
        # Segment using SAM
        detections = self._segment(image, detections, polygon_refinement=True)
        
        # Return the mask of the first detection (highest confidence)
        if detections:
            return detections[0].mask
        
        return None

    def register(self, K, rgb, depth, ob_mask=None, iteration=5):
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=iteration)
        self.is_initialized = True
        return pose
    
    def track(self, K, rgb, depth, iteration=2):
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call register first.")
        return self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
    
    def _handle_register(self, request):
        """Handle register command with automatic segmentation"""
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            labels = request.get('labels', ["object."])  # Default label
            threshold = request.get('threshold', 0.3)
            iteration = request.get('iteration', 5)
            
            logging.info("Received register command with automatic segmentation")
            
            # Perform automatic segmentation using Grounding DINO + SAM
            ob_mask = self._auto_segment(rgb, labels, threshold)
            
            if ob_mask is None:
                # Fallback: use depth-based mask
                logging.warning("Automatic segmentation failed, using depth-based mask")
                ob_mask = (depth > 0).astype(bool)
            else:
                ob_mask = (ob_mask > 0).astype(bool)
            
            # Execute register
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
    
    def execute_full_pipeline(self):
        reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
        for i in range(len(reader.color_files)):
            logging.info(f'i: {i}')
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            if i == 0:
                mask = reader.get_mask(0).astype(bool)
                pose = self.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask)
            else:
                pose = self.track(K=reader.K, rgb=color, depth=depth)
        
            os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

            os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{args.debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    def get_and_visualize_object_pose(self):
        K_color = np.array([
            [607.62, 0.00,  329.68],
            [0.00,  608.40, 243.36],
            [0.00,  0.00, 1.00]
        ])
        rgb_path = "/home/jszn/hewu/alg-product/FoundationPose/assets/zerith_rgb.png"
        depth_path = "/home/jszn/hewu/alg-product/FoundationPose/assets/zerith_depth.npy"
        for i in range(1000):
            logging.info(f'i: {i}')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            depth = np.load(depth_path) / 1000.0
            if i == 0:
                # Perform automatic segmentation using Grounding DINO + SAM
                labels = ['a black part']
                threshold = 0.3
                ob_mask = self._auto_segment(rgb, labels, threshold)
                
                if ob_mask is None:
                    logging.warning("Automatic segmentation failed, using depth-based mask")
                    ob_mask = (depth > 0).astype(bool)
                else:
                    ob_mask = (ob_mask > 0).astype(bool)
                pose = self.register(K=K_color, rgb=rgb, depth=depth, ob_mask=ob_mask)
            else:
                pose = self.track(K=K_color, rgb=rgb, depth=depth)
        
            os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{args.debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(K_color, img=rgb, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K_color, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

            os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{args.debug_dir}/track_vis/{i}.png', vis)

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

    def check_depth(self):
        rgb_path = "/home/jszn/hewu/alg-product/FoundationPose/assets/zerith_rgb.png"
        depth_path = "/home/jszn/hewu/alg-product/FoundationPose/assets/zerith_depth.npy"
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = np.load(depth_path) / 1000.0
        labels = ['a black part']
        threshold = 0.3
        ob_mask = self._auto_segment(rgb, labels, threshold)
        ob_mask = (ob_mask > 0).astype(bool)
        print(f"ob_mask", ob_mask)
        valid = (depth>=0.001) & (ob_mask>0)
        print(f"valid", valid.sum())

def main(args): 
    server = ZerithFoundationPoseServer(
        mesh_file=args.mesh_file,
        detector_id=args.detector_id,
        segmenter_id=args.segmenter_id
    )
    # server.start(port=args.zmq_port)
    server.get_and_visualize_object_pose()


if __name__ == '__main__':
    args = parse_args()
    main(args)
