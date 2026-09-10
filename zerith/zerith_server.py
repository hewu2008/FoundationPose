# encoding:utf8
import zmq
import pickle
import logging
import json
import gc
import time
import torch
import cv2
import os
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from zerith.zerith_mesh_utils import blue_z_plane_angle_deg


class ZerithServer:
    def __init__(self, pose_estimator, segmentation, save_dir='./debug_register'):
        self.pose_estimator = pose_estimator
        self.segmentation = segmentation
        self.is_initialized = False
        self._call_index = 0
        self._detection_index = 0
        self._all_parts_rgb = None
        self._all_parts_mask = None
        self._pending_detection_debug = None
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_overlay'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ob_in_cam'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'detections'), exist_ok=True)
    
    def _handle_ping(self, request):
        logging.debug("Received ping command")
        return {
            'status': 'success',
            'message': 'Server is running'
        }

    def _handle_detection(self, request):
        started = time.perf_counter()
        detection_index = self._detection_index
        # A new detection invalidates any image bundle left by the preceding
        # capture, even if this detection later fails.
        self._pending_detection_debug = None
        try:
            rgb_array = request['rgb'].astype(np.uint8)
            logging.info(
                "EVENT=DETECTION_START detection_index=%d image_shape=%s",
                detection_index,
                tuple(rgb_array.shape),
            )
            # A detection request starts a new RGB frame. Subsequent single
            # register calls accumulate their masks into one all-parts image.
            self._reset_all_parts_mask_overlay(rgb_array)
            rgb = Image.fromarray(rgb_array)
            try:
                self.segmentation.activate_detector()
                boxes = self.segmentation.detect_part(rgb)
            finally:
                # YOLO stays resident by default. Low-memory deployments can
                # opt into the legacy offload path explicitly.
                if not getattr(
                    self.segmentation, 'keep_detector_resident', False
                ):
                    self.segmentation.offload_detector()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            box_count = len(boxes or [])
            self._detection_index += 1
            detection_debug = self._save_detection_debug(
                detection_index, rgb_array, boxes or []
            )
            detection_ids = {
                int(box['detection_id'])
                for box in (boxes or [])
                if box.get('detection_id') is not None
            }
            if detection_debug is not None and len(detection_ids) == 1:
                self._pending_detection_debug = {
                    'detection_id': detection_ids.pop(),
                    'detection_boxes': detection_debug,
                }
            elif box_count:
                logging.warning(
                    "Detection debug image will not be returned: expected "
                    "one detection_id, got %s",
                    sorted(detection_ids),
                )
            detection_seconds = time.perf_counter() - started
            logging.info(
                "EVENT=DETECTION_DONE detection_index=%d instances=%d "
                "elapsed_seconds=%.6f",
                detection_index,
                box_count,
                detection_seconds,
            )

            if box_count == 0:
                return {
                    'status': 'error',
                    'message': 'No boxes detected'
                }
            else:
                return {
                    'status': 'success',
                    'boxes': boxes,
                    'timing': {
                        'total_seconds': float(detection_seconds),
                    },
                    'message': f'Detection successful, found {box_count} boxes'
                }   
        except Exception as e:
            logging.exception(
                "EVENT=ERROR phase=detection detection_index=%d "
                "error_type=%s",
                detection_index,
                type(e).__name__,
            )
            return {
                'status': 'error',
                'message': f'Detection failed: {str(e)}'
            }

    def _get_part_config(self, category_id=None, label=None):
        config = self.segmentation.get_category_config(
            category_id=category_id,
            label=label,
        )
        if config is None:
            raise ValueError(
                f"Unknown part: category_id={category_id!r}, label={label!r}"
            )
        return config

    def _category_sort_key(self, config):
        sorter = getattr(self.segmentation, 'category_sort_key', None)
        if callable(sorter):
            return (
                0,
                int(sorter(
                    category_id=config.get('id'),
                    label=config.get('label'),
                )),
                str(config.get('id', '')),
            )

        category_id = str(config.get('id', ''))
        suffix = category_id[3:] if category_id.startswith('cat') else ''
        if suffix.isdigit():
            return (1, int(suffix), category_id)
        return (2, category_id)

    @staticmethod
    def _normalize_mask(ob_mask, image_shape):
        if ob_mask is None:
            raise ValueError("Segmentation failed, no valid mask found")
        if isinstance(ob_mask, bytes):
            ob_mask = np.frombuffer(ob_mask, dtype=np.uint8).reshape(
                image_shape[:2]
            )
        ob_mask = (np.asarray(ob_mask) > 0).astype(bool)
        if ob_mask.shape != tuple(image_shape[:2]):
            raise ValueError(
                f"Mask shape {ob_mask.shape} does not match RGB {image_shape[:2]}"
            )
        if not ob_mask.any():
            raise ValueError("Segmentation returned an empty mask")
        return ob_mask

    def _register_from_mask(
        self,
        K,
        rgb,
        depth,
        config,
        ob_mask,
        iteration,
        frame_context=None,
        request_index=None,
    ):
        """Run the unchanged FoundationPose path for one prepared instance mask."""
        item_started = time.perf_counter()
        ob_mask = self._normalize_mask(ob_mask, rgb.shape)
        index = self._call_index
        self._call_index += 1

        ys, xs = np.where(ob_mask)
        mask_box = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ]
        mask_pixels = int(ob_mask.sum())
        category_id = str(config.get("id", "unknown"))
        logging.info(
            "EVENT=POSE_START category=%s request_index=%s debug_index=%d "
            "iteration=%d mask_box=%s mask_pixels=%d",
            category_id,
            request_index,
            index,
            int(iteration),
            mask_box,
            mask_pixels,
        )

        self._save_debug_images(index, rgb, ob_mask, depth)
        logging.info("Debug images saved to %s", self.save_dir)

        mesh_file = config["mesh_file"]
        self.pose_estimator.reset_object(mesh_file)
        estimation_started = time.perf_counter()
        try:
            pose = self.pose_estimator.register(
                K,
                rgb,
                depth,
                ob_mask,
                iteration=iteration,
                pose_config=config,
                frame_context=frame_context,
            )
        except torch.cuda.OutOfMemoryError:
            # Batch mode intentionally avoids empty_cache() between normal
            # instances.  If a fragmented allocator still reaches OOM, clear
            # once and retry the identical deterministic registration.
            logging.warning(
                "EVENT=OOM_RETRY category=%s request_index=%s debug_index=%d "
                "action=clear_cache_and_retry_once",
                category_id,
                request_index,
                index,
            )
            # If resident YOLO caused memory pressure, switch permanently to
            # the low-memory offload path for subsequent requests.
            if hasattr(self.segmentation, 'keep_detector_resident'):
                self.segmentation.keep_detector_resident = False
            offload_detector = getattr(
                self.segmentation, 'offload_detector', None
            )
            if callable(offload_detector):
                offload_detector()
            gc.collect()
            torch.cuda.empty_cache()
            pose = self.pose_estimator.register(
                K,
                rgb,
                depth,
                ob_mask,
                iteration=iteration,
                pose_config=config,
                frame_context=frame_context,
            )
        if hasattr(pose, "detach"):
            pose_array = pose.detach().cpu().numpy().reshape(4, 4)
        else:
            pose_array = np.asarray(pose).reshape(4, 4)
        estimation_seconds = time.perf_counter() - estimation_started
        logging.info(
            "EVENT=POSE_DONE category=%s request_index=%s debug_index=%d "
            "estimation_seconds=%.6f pose_4x4=%s",
            category_id,
            request_index,
            index,
            estimation_seconds,
            json.dumps(pose_array.tolist(), separators=(",", ":")),
        )

        pose_path = os.path.join(
            self.save_dir, 'ob_in_cam', f'{index}.txt'
        )
        np.savetxt(pose_path, pose_array)
        logging.info(
            "EVENT=POSE_SAVED category=%s request_index=%s debug_index=%d "
            "path=%s elapsed_seconds=%.6f",
            category_id,
            request_index,
            index,
            pose_path,
            time.perf_counter() - item_started,
        )
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        # Cat5 displays its authored CAD axes while its cuboid remains in the
        # OBB frame. Other categories retain their centered OBB axes. Return
        # the angle of the same blue +Z axis that the client visualizes.
        try:
            axis_to_origin = (
                None
                if category_id == 'cat5'
                else getattr(self.pose_estimator, 'to_origin', None)
            )
            blue_angle = blue_z_plane_angle_deg(
                K,
                pose,
                to_origin=axis_to_origin,
            )
        except ValueError as angle_error:
            # A +Z axis exactly aligned with the optical axis has no unique
            # image-plane direction.  Keep the valid 6D pose and report a
            # missing angle instead of failing the whole registration.
            logging.warning("Blue +Z plane angle unavailable: %s", angle_error)
            blue_angle = None
        return {
            'status': 'success',
            'pose': pose,
            'blue_z_plane_angle_deg': blue_angle,
            'mask_box': mask_box,
            'mask_pixels': mask_pixels,
            'debug_index': int(index),
            'pose_rerank': getattr(self.pose_estimator, 'last_rerank', None),
            'message': 'Registration successful',
        }

    def _handle_register(self, request):
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            label = request.get('label', None)
            category_id = request.get('category_id', None)
            box = request.get('box', None)
            threshold = request.get('threshold', 0.3)
            iteration = request.get('iteration', 5)
            debug_image_policy = request.get('debug_image_policy') or {}
            object_index = request.get('object_index')
            object_count = request.get('object_count')
            round_index = request.get('round_index')
            source_image_id = request.get('source_image_id')
            detection_set_id = request.get('detection_set_id')
            detection_id = request.get('detection_id')

            config = self._get_part_config(
                category_id=category_id,
                label=label,
            )

            logging.info("Received register command with automatic segmentation")

            ob_mask = self.segmentation.segment(rgb, label, box, threshold)

            logging.info("Segmentation completed")
            all_parts_debug = self._save_all_parts_mask_overlay(rgb, [ob_mask])
            response = self._register_from_mask(
                K,
                rgb,
                depth,
                config,
                ob_mask,
                iteration=iteration,
            )
            debug_images = self._build_final_register_debug_images(
                debug_image_policy=debug_image_policy,
                pending_detection_debug=self._pending_detection_debug,
                all_parts_debug=all_parts_debug,
                object_index=object_index,
                object_count=object_count,
                round_index=round_index,
                source_image_id=source_image_id,
                detection_set_id=detection_set_id,
                detection_id=detection_id,
            )
            if debug_images is not None:
                response['debug_images'] = debug_images
                if self._pending_detection_debug is not None:
                    response['detection_id'] = int(
                        self._pending_detection_debug['detection_id']
                    )
                self._pending_detection_debug = None
            return response
        except Exception as e:
            logging.exception(
                "EVENT=ERROR phase=register error_type=%s",
                type(e).__name__,
            )
            return {
                'status': 'error',
                'message': f'Registration failed: {str(e)}'
            }

    def _handle_register_batch(self, request):
        """Fetch N cached YOLO masks, then run the pose path in order."""
        started = time.perf_counter()
        # Consume the pending image exactly once. A failed or mismatched batch
        # must not make it available to a later capture.
        pending_detection_debug = self._pending_detection_debug
        self._pending_detection_debug = None
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            objects = request.get('objects')
            if not isinstance(objects, list) or not objects:
                raise ValueError("register_batch requires a non-empty objects list")

            logging.info(
                "EVENT=BATCH_START object_count=%d image_shape=%s",
                len(objects),
                tuple(np.asarray(rgb).shape),
            )

            configs = [
                self._get_part_config(
                    category_id=item.get('category_id'),
                    label=item.get('label'),
                )
                for item in objects
            ]
            segmentation_started = time.perf_counter()
            masks = self.segmentation.segment_many(rgb, objects)
            segmentation_seconds = time.perf_counter() - segmentation_started
            if len(masks) != len(objects):
                raise RuntimeError(
                    f"YOLO cache returned {len(masks)} masks for "
                    f"{len(objects)} objects"
                )

            registration_started = time.perf_counter()
            frame_started = time.perf_counter()
            prepare_frame = getattr(self.pose_estimator, 'prepare_frame', None)
            frame_context = (
                prepare_frame(K, rgb, depth)
                if callable(prepare_frame)
                else None
            )
            frame_preprocess_seconds = time.perf_counter() - frame_started
            logging.info(
                "EVENT=FRAME_PREPROCESS_DONE elapsed_seconds=%.6f "
                "mask_lookup_seconds=%.6f",
                frame_preprocess_seconds,
                segmentation_seconds,
            )

            ordered_objects = sorted(
                zip(range(len(objects)), objects, configs, masks),
                key=lambda entry: (
                    self._category_sort_key(entry[2]),
                    entry[0],
                ),
            )
            all_parts_debug = self._save_all_parts_mask_overlay(
                rgb,
                [entry[3] for entry in ordered_objects],
                reset=True,
            )
            results = []
            for request_index, item, config, ob_mask in ordered_objects:
                item_started = time.perf_counter()
                try:
                    result = self._register_from_mask(
                        K,
                        rgb,
                        depth,
                        config,
                        ob_mask,
                        iteration=int(item.get('iteration', 5)),
                        frame_context=frame_context,
                        request_index=request_index,
                    )
                except Exception as item_error:
                    logging.exception(
                        "EVENT=ERROR phase=pose category=%s request_index=%d "
                        "error_type=%s",
                        config.get('id', 'unknown'),
                        request_index,
                        type(item_error).__name__,
                    )
                    result = {
                        'status': 'error',
                        'message': f'Registration failed: {item_error}',
                    }
                result['request_index'] = int(request_index)
                result['category_id'] = str(config['id'])
                result['label'] = str(config['label'])
                result['elapsed_seconds'] = float(
                    time.perf_counter() - item_started
                )
                results.append(result)

            registration_seconds = (
                time.perf_counter() - registration_started
            )
            successful = sum(
                result.get('status') == 'success' for result in results
            )
            timing = {
                'segmentation_seconds': float(segmentation_seconds),
                'frame_preprocess_seconds': float(
                    frame_preprocess_seconds
                ),
                'registration_seconds': float(registration_seconds),
                'total_seconds': float(time.perf_counter() - started),
            }
            logging.info(
                "EVENT=BATCH_DONE object_count=%d successful=%d failed=%d "
                "mask_lookup_seconds=%.6f frame_preprocess_seconds=%.6f "
                "registration_seconds=%.6f total_seconds=%.6f",
                len(results),
                successful,
                len(results) - successful,
                timing['segmentation_seconds'],
                timing['frame_preprocess_seconds'],
                timing['registration_seconds'],
                timing['total_seconds'],
            )
            response = {
                'status': 'success',
                'results': results,
                'successful': int(successful),
                'failed': int(len(results) - successful),
                'timing': timing,
                'message': (
                    f'Batch registration completed: {successful}/'
                    f'{len(results)} successful'
                ),
            }
            debug_images = self._build_batch_debug_images(
                pending_detection_debug,
                objects,
                all_parts_debug,
            )
            if debug_images is not None:
                response['detection_id'] = int(
                    pending_detection_debug['detection_id']
                )
                response['debug_images'] = debug_images
            return response
        except Exception as e:
            logging.exception(
                "EVENT=ERROR phase=batch_register error_type=%s",
                type(e).__name__,
            )
            return {
                'status': 'error',
                'message': f'Batch registration failed: {str(e)}',
            }

    def _handle_track(self, request):
        try:
            K = request['K']
            rgb = request['rgb']
            depth = request['depth']
            iteration = request.get('iteration', 2)

            logging.info("Received track command")

            pose = self.pose_estimator.track(K, rgb, depth, iteration)
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose)

            return {
                'status': 'success',
                'pose': pose,
                'message': 'Tracking successful'
            }
        except Exception as e:
            logging.exception(
                "EVENT=ERROR phase=track error_type=%s",
                type(e).__name__,
            )
            return {
                'status': 'error',
                'message': f'Tracking failed: {str(e)}'
            }

    def _handle_unknown(self, command):
        logging.warning(f"Unknown command: {command}")
        return {
            'status': 'error',
            'message': f'Unknown command: {command}'
        }

    def _save_debug_images(self, index, rgb, ob_mask, depth=None):
        try:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_dir, 'rgb', f'{index}.png'), rgb_bgr)
            if depth is not None:
                depth_array = np.asarray(depth)
                if depth_array.shape != ob_mask.shape:
                    raise ValueError(
                        f"Depth shape {depth_array.shape} does not match "
                        f"mask shape {ob_mask.shape}"
                    )
                # Incoming depth is expressed in metres. Store a lossless
                # 16-bit millimetre PNG and clear pixels outside this part so
                # every file contains only its corresponding component.
                part_depth_mm = np.zeros(ob_mask.shape, dtype=np.uint16)
                valid = ob_mask & np.isfinite(depth_array) & (depth_array > 0)
                part_depth_mm[valid] = np.clip(
                    np.rint(depth_array[valid] * 1000.0),
                    0,
                    np.iinfo(np.uint16).max,
                ).astype(np.uint16)
                depth_path = os.path.join(
                    self.save_dir, 'depth', f'{index}.png'
                )
                if not cv2.imwrite(depth_path, part_depth_mm):
                    raise OSError(f"Failed to write depth image: {depth_path}")
            cv2.imwrite(
                os.path.join(self.save_dir, 'mask', f'{index}.png'),
                ob_mask.astype(np.uint8) * 255,
            )

            mask_vis = rgb.copy()
            mask_vis[ob_mask] = [0, 255, 0]
            mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_dir, 'mask_overlay', f'{index}.png'), mask_vis_bgr)

            logging.info(f"Debug images saved for index {index}")
        except Exception as save_e:
            logging.warning(f"Failed to save debug images: {str(save_e)}")    

    def _reset_all_parts_mask_overlay(self, rgb):
        """Start accumulation for a newly detected RGB frame."""
        rgb_array = np.asarray(rgb, dtype=np.uint8)
        self._all_parts_rgb = rgb_array.copy()
        self._all_parts_mask = np.zeros(rgb_array.shape[:2], dtype=bool)

    def _save_all_parts_mask_overlay(self, rgb, masks, reset=False):
        """Accumulate part masks and save them as solid green on one image."""
        try:
            rgb_array = np.asarray(rgb, dtype=np.uint8)
            frame_changed = (
                self._all_parts_rgb is None
                or self._all_parts_rgb.shape != rgb_array.shape
                or not np.array_equal(self._all_parts_rgb, rgb_array)
            )
            if reset or frame_changed:
                self._reset_all_parts_mask_overlay(rgb_array)

            mask_count = 0
            for ob_mask in masks:
                try:
                    mask = self._normalize_mask(ob_mask, rgb_array.shape)
                except ValueError as mask_error:
                    logging.warning(
                        "Skipping invalid mask in all-parts overlay: %s",
                        mask_error,
                    )
                    continue
                self._all_parts_mask |= mask
                mask_count += 1

            overlay = self._all_parts_rgb.copy()
            overlay[self._all_parts_mask] = [0, 255, 0]
            overlay_path = os.path.join(
                self.save_dir, 'mask_overlay', 'all_parts.png'
            )
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            encoded_ok, encoded_png = cv2.imencode('.png', overlay_bgr)
            if not encoded_ok:
                raise OSError(
                    "Failed to encode all-parts mask overlay as PNG"
                )
            image_bytes = encoded_png.tobytes()
            with open(overlay_path, 'wb') as image_file:
                image_file.write(image_bytes)
            logging.info(
                "EVENT=ALL_PARTS_MASK_OVERLAY_SAVED new_masks=%d "
                "total_pixels=%d path=%s",
                mask_count,
                int(self._all_parts_mask.sum()),
                overlay_path,
            )
            return {
                'filename': 'all_parts.png',
                'content_type': 'image/png',
                'data': image_bytes,
            }
        except Exception as save_e:
            logging.warning(
                "Failed to save all-parts mask overlay: %s", save_e
            )
            return None

    def _save_detection_debug(self, index, rgb, boxes):
        """Persist a visualization of the raw 2D detector result."""
        try:
            vis = Image.fromarray(rgb.astype(np.uint8)).copy()
            draw = ImageDraw.Draw(vis)
            for box in boxes:
                xyxy = tuple(float(box[key]) for key in ('x1', 'y1', 'x2', 'y2'))
                draw.rectangle(xyxy, outline='red', width=3)
                draw.text(
                    (xyxy[0], max(0, xyxy[1] - 14)),
                    str(box.get('category_id', box.get('label', 'part'))),
                    fill='red',
                )
            filename = f'{index}_boxes.png'   ##### boxes
            image_buffer = BytesIO()
            vis.save(image_buffer, format='PNG')
            image_bytes = image_buffer.getvalue()
            image_path = os.path.join(
                self.save_dir, 'detections', filename
            )
            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)
            return {
                'filename': filename,
                'content_type': 'image/png',
                'data': image_bytes,
            }
        except Exception as save_e:
            logging.warning("Failed to save detection debug: %s", save_e)
            return None

    @staticmethod
    def _build_batch_debug_images(
        pending_detection_debug,
        objects,
        all_parts_debug,
    ):
        """Return the two PNGs only when they belong to this exact batch."""
        if pending_detection_debug is None or all_parts_debug is None:
            return None

        supplied_ids = []
        for item in objects:
            detection_id = item.get('detection_id')
            if detection_id is None:
                logging.warning(
                    "Debug images omitted: register_batch object has no "
                    "detection_id"
                )
                return None
            try:
                supplied_ids.append(int(detection_id))
            except (TypeError, ValueError):
                logging.warning(
                    "Debug images omitted: invalid detection_id=%r",
                    detection_id,
                )
                return None

        expected_id = int(pending_detection_debug['detection_id'])
        if not supplied_ids or set(supplied_ids) != {expected_id}:
            logging.warning(
                "Debug images omitted: batch detection_ids=%s do not match "
                "pending detection_id=%d",
                sorted(set(supplied_ids)),
                expected_id,
            )
            return None

        all_parts_response = dict(all_parts_debug)
        all_parts_response['filename'] = f'{expected_id}_all_parts.png'  ####mask
        return {
            'detection_boxes': pending_detection_debug['detection_boxes'],
            'all_parts': all_parts_response,
        }

    @staticmethod
    def _is_final_register_object(object_index, object_count):
        try:
            return int(object_index) == int(object_count) - 1 and int(object_count) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _debug_policy_enabled(policy):
        if not isinstance(policy, dict):
            return False
        if not bool(policy.get('enabled', False)):
            return False
        scope = str(policy.get('scope') or 'first_input_only').strip().lower()
        timing = str(
            policy.get('return_timing') or policy.get('return_when') or ''
        ).strip().lower()
        mode = str(policy.get('return_mode') or '').strip().lower()
        return (
            scope == 'first_input_only'
            and timing in {'after_all_objects_segmented', 'all_objects_segmented'}
            and mode in {'combined_final', 'final_combined', ''}
        )

    def _build_final_register_debug_images(
        self,
        *,
        debug_image_policy,
        pending_detection_debug,
        all_parts_debug,
        object_index,
        object_count,
        round_index=None,
        source_image_id=None,
        detection_set_id=None,
        detection_id=None,
    ):
        if not self._debug_policy_enabled(debug_image_policy):
            return None
        if not self._is_final_register_object(object_index, object_count):
            return None
        if pending_detection_debug is None or all_parts_debug is None:
            logging.warning(
                "Final register debug images omitted: detection_debug=%s "
                "all_parts_debug=%s",
                pending_detection_debug is not None,
                all_parts_debug is not None,
            )
            return None

        if detection_id is not None:
            try:
                expected_id = int(pending_detection_debug['detection_id'])
                supplied_id = int(detection_id)
            except (TypeError, ValueError):
                logging.warning(
                    "Final register debug images omitted: invalid detection_id=%r",
                    detection_id,
                )
                return None
            if supplied_id != expected_id:
                logging.warning(
                    "Final register debug images omitted: detection_id=%d does "
                    "not match pending detection_id=%d",
                    supplied_id,
                    expected_id,
                )
                return None

        all_parts_response = dict(all_parts_debug)
        all_parts_response['filename'] = 'first_all_parts.png'
        detection_boxes_response = dict(pending_detection_debug['detection_boxes'])
        detection_boxes_response['filename'] = 'first_detection_boxes.png'

        return {
            'sets': [
                {
                    'round_index': round_index,
                    'stage': 'first_input_complete',
                    'source_image_id': source_image_id,
                    'detection_set_id': detection_set_id,
                    'images': {
                        'detection_boxes': detection_boxes_response,
                        'all_parts': all_parts_response,
                    },
                }
            ]
        }
     
    def _process_request(self, message):
        try:
            request = pickle.loads(message)
            command = request.get('command', '')

            handlers = {
                'ping': self._handle_ping,
                'detection': self._handle_detection,
                'register': self._handle_register,
                'register_batch': self._handle_register_batch,
                'track': self._handle_track
            }

            handler = handlers.get(command, lambda req: self._handle_unknown(command))
            return handler(request)

        except Exception as e:
            logging.exception(
                "EVENT=ERROR phase=request_processing error_type=%s",
                type(e).__name__,
            )
            return {
                'status': 'error',
                'message': str(e)
            }

    def start(self, port=5555, startup_elapsed_seconds=None):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        logging.info(
            "EVENT=SERVER_READY port=%d startup_elapsed_seconds=%s",
            port,
            (
                f"{float(startup_elapsed_seconds):.6f}"
                if startup_elapsed_seconds is not None
                else "unknown"
            ),
        )

        try:
            while True:
                message = socket.recv()
                response = self._process_request(message)
                socket.send(pickle.dumps(response))

        except KeyboardInterrupt:
            logging.info("EVENT=SERVER_STOP reason=keyboard_interrupt")
        finally:
            socket.close()
            context.term()
