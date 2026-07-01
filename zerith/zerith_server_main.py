# encoding:utf8
import argparse
import logging
import numpy as np
import os
import cv2
import imageio

from zerith_segmentation import ZerithSegmentation
from zerith_pose_estimator import ZerithPoseEstimator
from zerith_server import ZerithServer
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
from Utils import set_logging_format, set_seed
from datareader import YcbineoatReader
from Utils import draw_posed_3d_box, draw_xyz_axis


def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Server')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the mesh file (OBJ format)')
    parser.add_argument('--zmq_port', type=int, default=5555, help='ZMQ server port')
    parser.add_argument('--detector_id', type=str, default="IDEA-Research/grounding-dino-tiny", help='Grounding DINO model ID')
    parser.add_argument('--segmenter_id', type=str, default="facebook/sam-vit-base", help='Segment Anything model ID')
    parser.add_argument('--test_scene_dir', type=str, default=None, help='Path to test scene directory')
    parser.add_argument('--debug_dir', type=str, default='./debug', help='Directory to save debug outputs')
    return parser.parse_args()


def create_pose_estimator(mesh_file, debug_dir):
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    logging.info("Load scorer, refiner and glctx successfully")

    return ZerithPoseEstimator(
        mesh_file=mesh_file,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir=debug_dir
    )


def create_segmentation(detector_id, segmenter_id):
    return ZerithSegmentation(
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )


def execute_full_pipeline(server, test_scene_dir, debug_dir, to_origin):
    if test_scene_dir is None:
        logging.warning("test_scene_dir not provided, skipping pipeline execution")
        return

    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        logging.info(f'Processing frame {i}/{len(reader.color_files)}')

        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = server.pose_estimator.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask)
        else:
            pose = server.pose_estimator.track(K=reader.K, rgb=color, depth=depth)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=server.pose_estimator.bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('FoundationPose', vis[..., ::-1])
        cv2.waitKey(1)

        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


def main(args):
    pose_estimator = create_pose_estimator(args.mesh_file, args.debug_dir)
    segmentation = create_segmentation(detector_id=args.detector_id, segmenter_id=args.segmenter_id)
    server = ZerithServer(pose_estimator=pose_estimator, segmentation=segmentation, save_dir=args.debug_dir)
    server.start(port=args.zmq_port)


if __name__ == '__main__':
    args = parse_args()
    main(args)
