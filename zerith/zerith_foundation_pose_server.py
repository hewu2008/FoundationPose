# encoding:utf8
import argparse
import trimesh
import logging
import numpy as np
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
from estimater import FoundationPose

def parse_args():
    parser = argparse.ArgumentParser(description='Zerith FoundationPose Server')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the mesh file (OBJ format)')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
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
            debug_dir=args.debug_dir, 
            debug=0, 
            glctx=self.glctx
        )
        logging.info(f"Load estimator successfully")

    def register(self, K, rgb, depth, ob_mask, iteration=5):
        return self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=iteration)
    
    def track(self, K, rgb, depth, iteration=2):
        return self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
    
    def run_scene(self):
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

    def start(self):
        logging.info(f"Starting server with mesh file: {self.mesh_file}")


def main(args): 
    server = ZerithFoundationPoseServer(mesh_file=args.mesh_file)
    server.run_scene()


if __name__ == '__main__':
    args = parse_args()
    main(args)
