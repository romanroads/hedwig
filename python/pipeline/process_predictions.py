import timeit
import numpy as np
from shapely import geometry
import logging

from detectron2.data import MetadataCatalog

from pipeline.pipeline import Pipeline
from utils import get_scaled_mask, calculate_principal_axis
from constant_values import *


class ProcessPredictions(Pipeline):
    def __init__(self, width_tracking, height_tracking, width_detection, height_detection,
                 scale_width_from_det_to_track, scale_height_from_det_to_track,
                 scale_width_from_det_to_output, scale_height_from_det_to_output, thresh_det):
        category_ids = list(MAP_CATEGORY_ID_TO_NAME.keys())
        category_ids.sort()
        MetadataCatalog.get(TRAINING_DATASET_NAME).set(thing_classes=[MAP_CATEGORY_ID_TO_NAME[category_id]
                                                                 for category_id in category_ids])
        self.car_metadata = MetadataCatalog.get(TRAINING_DATASET_NAME)

        self.width_tracking = width_tracking
        self.height_tracking = height_tracking
        self.width_detection = width_detection
        self.height_detection = height_detection
        self.scale_width_from_det_to_track = scale_width_from_det_to_track
        self.scale_height_from_det_to_track = scale_height_from_det_to_track
        self.scale_width_from_det_to_output = scale_width_from_det_to_output
        self.scale_height_from_det_to_output = scale_height_from_det_to_output

        # Note: the grid is defined in the detection space, rather the tracking space
        self.num_row = 2
        self.num_col = 1
        self.num_grid_cells = self.num_row * self.num_col
        self.width_interval = self.width_detection / self.num_col
        self.height_interval = self.height_detection / self.num_row

        self.is_stab_object_injected = False

        if len(thresh_det) > 0:
            threshs = thresh_det.split(",")
            n_thresh = len(threshs)
            logging.info(
                "process_predictions: %s thresholds are provided for detecting different objects" % n_thresh)
            for i in range(n_thresh):
                thresh_obj = threshs[i]
                obj, threshold = thresh_obj.split(":")
                threshold = float(threshold)
                logging.info("process_predictions: object %s has desired detection threshold %.1f" % (obj, threshold))
                if obj in THRESHOLD_DETECTION_PRECISION:
                    logging.info("process_predictions: object %s (threshold = %.1f) accepted" % (obj, threshold))
                    THRESHOLD_DETECTION_PRECISION[obj] = threshold

        super().__init__("ProcessPredictions")

    def map(self, data):
        start_time = timeit.default_timer()

        if PREDICTIONS_NAME in data:
            predictions = data[PREDICTIONS_NAME]

            if predictions is not None:
                self.process_predictions(predictions, data)

        self.timer += timeit.default_timer() - start_time
        return data

    def process_predictions(self, predictions, data):
        detected_object = predictions['instances'].to("cpu")
        masks = detected_object.pred_masks.numpy()
        boxes = detected_object.pred_boxes if detected_object.has("pred_boxes") else None
        scores = detected_object.scores if detected_object.has("scores") else None
        classes = detected_object.pred_classes if detected_object.has("pred_classes") else None
        feature_points = detected_object.pred_keypoints.numpy() if detected_object.has("pred_keypoints") else None
        num_detected_objects = len(masks)

        processed_predictions = {}

        bool_mask_tracking_combo = None

        for i in range(num_detected_objects):
            agent_class = self.car_metadata.get("thing_classes", None)[classes[i]]

            precision_prediction = scores[i]

            if precision_prediction < THRESHOLD_DETECTION_PRECISION[agent_class]:
                continue

            is_agent_dynamic = agent_class in CATEGORY_NAMES_FOR_DYNAMIC_OBJECTS

            # Note: original mask from DNN output is boolean matrix
            original_mask = masks[i]
            color_mask_detection = np.uint8(255) * original_mask
            scaled_mask = get_scaled_mask(original_mask, self.width_tracking, self.height_tracking)

            if bool_mask_tracking_combo is None:
                bool_mask_tracking_combo = scaled_mask
            else:
                bool_mask_tracking_combo |= scaled_mask

            color_mask_tracking = np.uint8(255) * scaled_mask

            larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1, \
                smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector, \
                offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4 = \
                calculate_principal_axis(color_mask_detection, self.scale_width_from_det_to_track,
                                         self.scale_height_from_det_to_track)

            polygon_length = np.linalg.norm(larger_principal_axis_point_1 - larger_principal_axis_point_2)
            polygon_width = np.linalg.norm(smaller_principal_axis_point_1 - smaller_principal_axis_point_2)

            x1, y1, x2, y2 = boxes.tensor[i]
            x_c, y_c = (x1 + x2) * 0.5, (y1 + y2) * 0.5

            index_col = int(x_c / self.width_interval)
            index_row = int(y_c / self.height_interval)
            index_grid_cell = index_row * self.num_col + index_col

            x_c *= self.scale_width_from_det_to_track
            y_c *= self.scale_height_from_det_to_track

            box_center = np.array([x_c, y_c])
            bbox_width = np.abs(x1 - x2) * self.scale_width_from_det_to_track
            bbox_height = np.abs(y1 - y2) * self.scale_height_from_det_to_track

            # Note: polygon boundary
            list_coordinates = [box_center + offset_corner_1, box_center + offset_corner_2,
                                box_center + offset_corner_3, box_center + offset_corner_4]

            # Note: feature points
            if feature_points is not None:
                feature_point = feature_points[i]
                feature_point[:, 0] *= self.scale_width_from_det_to_track
                feature_point[:, 1] *= self.scale_height_from_det_to_track
            else:
                feature_point = None

            processed_predictions[i] = {
                PROCESSED_PREDICTIONS_CENTER_NAME: box_center,
                PROCESSED_PREDICTIONS_LENGTH_NAME: polygon_length,
                PROCESSED_PREDICTIONS_WIDTH_NAME: polygon_width,
                PROCESSED_PREDICTIONS_BOX_WIDTH_NAME: bbox_width,
                PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME: bbox_height,
                PROCESSED_PREDICTIONS_POLY_POINTS_NAME: list_coordinates,
                PROCESSED_PREDICTIONS_POLYGON_NAME: geometry.Polygon(list_coordinates),
                PROCESSED_PREDICTIONS_MASK_DET_NAME: color_mask_detection,
                PROCESSED_PREDICTIONS_MASK_TRACK_NAME: color_mask_tracking,
                PROCESSED_PREDICTIONS_POS_INDEX_NAME: [index_grid_cell, index_col, index_row],
                PROCESSED_PREDICTIONS_DYNAMIC_NAME: is_agent_dynamic,
                PROCESSED_PREDICTIONS_CLASS_NAME: agent_class,
                PROCESSED_PREDICTIONS_EIGENVECTOR_NAME:
                    [larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1,
                     smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector,
                     offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4],
                PROCESSED_PREDICTIONS_FEATURE_POINTS_NAME: feature_point,
                PROCESSED_PREDICTIONS_CONFIDENCE_NAME: precision_prediction
            }

        # Note: need more thoughts on this, here we insert manually created stablization zones temporarily into
        # detection object lists, later hopefully this will be automatically done by DNN, not by human
        if STABLIZATION_ZONE_NAME in data and not self.is_stab_object_injected:
            list_stab_objects = data[STABLIZATION_ZONE_NAME]
            num_stab_objects = sum([0 if len(s) <= 0 else 1 for s in list_stab_objects])

            if num_stab_objects > 0:

                width_authentic_resolution_roi, \
                height_authentic_resolution_roi, \
                width_starting_point_authentic_resolution_roi, \
                height_starting_point_authentic_resolution_roi, _, _ = \
                    data[DIMENSIONS_NAME]

                ori_image_width, ori_image_height = data[ORI_FRAME_DIMEN_NAME]

                # Note: no overlap with DNN detected objects
                index_stab_obj_to_submit = num_detected_objects
                count_stab_obj = 0
                index_stab_obj = 0

                while count_stab_obj < num_stab_objects:
                    stab_obj = list_stab_objects[index_stab_obj]

                    if len(stab_obj) <= 0:
                        index_stab_obj += 1
                        continue

                    bound_points = []
                    for i in range(len(stab_obj)):
                        p = stab_obj[i]
                        w, h = p
                        w, h = w * ori_image_width, h * ori_image_height
                        w, h = w - width_starting_point_authentic_resolution_roi,\
                            h - height_starting_point_authentic_resolution_roi
                        w *= self.width_tracking / width_authentic_resolution_roi
                        h *= self.height_tracking / height_authentic_resolution_roi
                        bound_points.append((w, h))

                    poly = geometry.Polygon(bound_points)

                    bound_points = np.array(bound_points).astype(np.float32)

                    # Note: take a mean along 0th axis, 1st axis width and height, bad idea....
                    box_center_min = np.min(bound_points, axis=0)
                    box_center_max = np.max(bound_points, axis=0)
                    box_center = (box_center_min + box_center_max) * 0.5

                    box = poly.minimum_rotated_rectangle
                    xs, ys = box.exterior.coords.xy

                    # get length of bounding box edges
                    edge_length = (geometry.Point(xs[0], ys[0]).distance(geometry.Point(xs[1], ys[1])),
                                   geometry.Point(xs[1], ys[1]).distance(geometry.Point(xs[2], ys[2])))

                    polygon_length = max(edge_length)
                    polygon_width = min(edge_length)
                    bbox_height = np.abs(poly.bounds[1] - poly.bounds[3])
                    bbox_width = np.abs(poly.bounds[0] - poly.bounds[2])

                    color_mask_detection = np.zeros((self.height_detection, self.width_detection), dtype=np.uint8)
                    color_mask_tracking = np.zeros((self.height_tracking, self.width_tracking), dtype=np.uint8)

                    color_mask_tracking[int(poly.bounds[1]):
                                        int(poly.bounds[3]),
                                        int(poly.bounds[0]):
                                        int(poly.bounds[2])
                                        ] = 255

                    h_ratio_track_to_det = 1. * self.height_detection / self.height_tracking
                    w_ratio_track_to_det = 1. * self.width_detection / self.width_tracking

                    color_mask_detection[int(poly.bounds[1] * h_ratio_track_to_det):
                                         int(poly.bounds[3] * h_ratio_track_to_det),
                                         int(poly.bounds[0] * w_ratio_track_to_det):
                                         int(poly.bounds[2] * w_ratio_track_to_det)
                                         ] = 255

                    index_col = int(box_center[0] * w_ratio_track_to_det / self.width_interval)
                    index_row = int(box_center[1] * h_ratio_track_to_det / self.height_interval)
                    index_grid_cell = index_row * self.num_col + index_col

                    # Note: we set this stablization zone to be class of Road Features, static objects
                    agent_class = "RoadFeatures"
                    precision_prediction = 1.
                    is_agent_dynamic = agent_class in CATEGORY_NAMES_FOR_DYNAMIC_OBJECTS

                    larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1, \
                    smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector, \
                    offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4 = \
                        calculate_principal_axis(color_mask_detection, self.scale_width_from_det_to_track,
                                                 self.scale_height_from_det_to_track)

                    feature_point = None

                    # Note: polygon boundary
                    list_coordinates = [box_center + offset_corner_1, box_center + offset_corner_2,
                                        box_center + offset_corner_3, box_center + offset_corner_4]

                    processed_predictions[index_stab_obj_to_submit] = {
                       PROCESSED_PREDICTIONS_CENTER_NAME: box_center,
                        PROCESSED_PREDICTIONS_LENGTH_NAME: polygon_length,
                        PROCESSED_PREDICTIONS_WIDTH_NAME: polygon_width,
                        PROCESSED_PREDICTIONS_BOX_WIDTH_NAME: bbox_width,
                        PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME: bbox_height,
                        PROCESSED_PREDICTIONS_POLY_POINTS_NAME: list_coordinates,
                        PROCESSED_PREDICTIONS_POLYGON_NAME: geometry.Polygon(list_coordinates),
                        PROCESSED_PREDICTIONS_MASK_DET_NAME: color_mask_detection,
                        PROCESSED_PREDICTIONS_MASK_TRACK_NAME: color_mask_tracking,
                        PROCESSED_PREDICTIONS_POS_INDEX_NAME: [index_grid_cell, index_col, index_row],
                        PROCESSED_PREDICTIONS_DYNAMIC_NAME: is_agent_dynamic,
                        PROCESSED_PREDICTIONS_CLASS_NAME: agent_class,
                        PROCESSED_PREDICTIONS_EIGENVECTOR_NAME:
                            [larger_principal_axis_point_1, larger_principal_axis_point_2,
                             smaller_principal_axis_point_1,
                             smaller_principal_axis_point_2, larger_principal_axis_vector,
                             smaller_principal_axis_vector,
                             offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4],
                        PROCESSED_PREDICTIONS_FEATURE_POINTS_NAME: feature_point,
                        PROCESSED_PREDICTIONS_CONFIDENCE_NAME: precision_prediction
                    }

                    count_stab_obj += 1
                    index_stab_obj += 1
                    index_stab_obj_to_submit += 1

            self.is_stab_object_injected = True

        data[PROCESSED_PREDICTIONS_NAME] = processed_predictions

        if feature_points is not None:
            data[PROCESSED_PREDICTIONS_FEATURE_POINTS_COMBO_NAME] = feature_points
            data[PROCESSED_PREDICTIONS_SCORES_COMBO_NAME] = scores

        data[PROCESSED_PREDICTIONS_MASK_COMBO_NAME] = bool_mask_tracking_combo

