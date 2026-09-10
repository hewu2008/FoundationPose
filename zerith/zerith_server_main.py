# encoding:utf8
import argparse
import logging
import os
from pathlib import Path
import time
import nvdiffrast.torch as dr

from zerith_yolo_segmentation import (
    DEFAULT_PARTS_CONFIG,
    DEFAULT_YOLO_WEIGHTS,
    ZerithYoloSegmentation,
    load_yolo_part_configs,
)
from zerith_pose_estimator import ZerithPoseEstimator
from zerith_server import ZerithServer
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor

try:
    from zerith.zerith_runtime_logging import configure_runtime_logging
except ImportError:  # Direct execution via zerith/zerith_server_main.py.
    from zerith_runtime_logging import configure_runtime_logging


DEFAULT_RUNTIME_LOG = (
    Path(__file__).resolve().parents[1] / "runtime" / "zerith_server.log"
)


def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Server')
    parser.add_argument('--mesh_file', type=str, default=None, help='Optional initial mesh; defaults to the first enabled part')
    parser.add_argument('--parts_config', type=str, default=str(DEFAULT_PARTS_CONFIG), help='JSON part and mesh configuration')
    parser.add_argument('--zmq_port', type=int, default=5555, help='ZMQ server port')
    parser.add_argument('--yolo_weights', type=str, default=str(DEFAULT_YOLO_WEIGHTS), help='YOLO26 instance-segmentation weights')
    parser.add_argument('--yolo_confidence', type=float, default=0.6, help='YOLO confidence threshold')
    parser.add_argument('--yolo_iou', type=float, default=0.7, help='YOLO NMS IoU threshold')
    parser.add_argument('--yolo_device', type=str, default=None, help='YOLO inference device; defaults to cuda:0 when available')
    parser.add_argument(
        '--offload_yolo',
        action='store_true',
        help='Move YOLO back to CPU after detection (slower, for low VRAM only)',
    )
    parser.add_argument(
        '--lazy_mesh_cache',
        action='store_true',
        help='Build mesh states on first use instead of during server startup',
    )
    parser.add_argument('--debug_dir', type=str, default='./debug', help='Directory to save debug outputs')
    parser.add_argument(
        '--log_file',
        type=str,
        default=str(DEFAULT_RUNTIME_LOG),
        help='Append timestamped server output to this runtime log file',
    )
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


def create_segmentation(
    yolo_weights,
    parts_config,
    confidence,
    iou_threshold,
    device=None,
    keep_detector_resident=True,
):
    return ZerithYoloSegmentation(
        weight_path=yolo_weights,
        parts_config=parts_config,
        confidence=confidence,
        iou_threshold=iou_threshold,
        device=device,
        keep_detector_resident=keep_detector_resident,
    )


def main(args):
    server_started = time.perf_counter()
    log_path = configure_runtime_logging(args.log_file)
    logging.info(
        "EVENT=SERVER_START pid=%d debug_dir=%s log_file=%s",
        os.getpid(),
        args.debug_dir,
        log_path,
    )
    try:
        model_started = time.perf_counter()
        logging.info(
            "EVENT=MODEL_LOAD_START components=scorer,refiner,glctx,cad,yolo"
        )
        parts = load_yolo_part_configs(args.parts_config)
        initial_mesh = args.mesh_file or parts[0]["mesh_file"]
        pose_estimator = create_pose_estimator(initial_mesh, args.debug_dir)
        if not args.lazy_mesh_cache:
            pose_estimator.preload_objects(
                part["mesh_file"] for part in parts
            )
        segmentation = create_segmentation(
            yolo_weights=args.yolo_weights,
            parts_config=args.parts_config,
            confidence=args.yolo_confidence,
            iou_threshold=args.yolo_iou,
            device=args.yolo_device,
            keep_detector_resident=not args.offload_yolo,
        )
        logging.info(
            "EVENT=MODEL_LOAD_DONE part_count=%d elapsed_seconds=%.6f",
            len(parts),
            time.perf_counter() - model_started,
        )
        server = ZerithServer(
            pose_estimator=pose_estimator,
            segmentation=segmentation,
            save_dir=args.debug_dir,
        )
        server.start(
            port=args.zmq_port,
            startup_elapsed_seconds=time.perf_counter() - server_started,
        )
    except Exception as error:
        logging.exception(
            "EVENT=ERROR phase=server_startup_or_run error_type=%s",
            type(error).__name__,
        )
        raise


if __name__ == '__main__':
    args = parse_args()
    main(args)
