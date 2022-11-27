from shapely import geometry
import sys
import timeit
import logging

from object_tracker import ObjectTracker
from utils import compute_eigenaxis_points
from constant_values import *


class StaticObjectTracker(ObjectTracker):
    global_static_object_id = 0
    global_static_agents = []
    global_static_lane_marker_width = 10.  # in units of pixels in tracking space
    global_static_lane_width = None  # in units of pixels in tracking space
    global_static_lane_width_min = sys.maxsize
    global_static_max_num_lane_marker = 21

    def __init__(self, list_occ, frame_gray, mask_detection, mask_tracking, center, polygon_length, polygon_width,
                 bbox_width, bbox_height,
                 fps,
                 agent_class, num_row, num_col, frame_counter, feature_detector="gftt", frame_color=None):
        self.is_initialized = False
        self.obj_id = StaticObjectTracker.global_static_object_id
        self.duration_lost = -1
        StaticObjectTracker.global_static_object_id += 1
        self.agent_class = agent_class

        self.color_code = (148, 0, 211)
        self.color_code_larger_principal_axis = (254, 80, 0)
        self.color_code_smaller_principal_axis = (116, 209, 234)
        self.thickness = 4

        # self.name = "%s%d" % (self.agent_class, self.obj_id)
        self.name = "%d" % self.obj_id
        self.display_name = self.name

        self.fps = fps
        self.frame_counter = frame_counter
        self.num_updates_from_detection = 0
        self.num_updates_from_tracking = 0

        self.index_to_update = 0
        self.period_to_update = 5

        self.mask_detection_space = mask_detection
        self.mask_tracking_space = mask_tracking
        self.scale_width_det_to_tracking = mask_tracking.shape[1] / mask_detection.shape[1]
        self.scale_height_det_to_tracking = mask_tracking.shape[0] / mask_detection.shape[0]

        self.larger_principal_axis_point_1 = np.array([0., 0.])
        self.larger_principal_axis_point_2 = np.array([0., 0.])
        self.smaller_principal_axis_point_1 = np.array([0., 0.])
        self.smaller_principal_axis_point_2 = np.array([0., 0.])
        self.larger_principal_axis_vector = np.array([0., 0.])
        self.smaller_principal_axis_vector = np.array([0., 0.])
        self.offset_corner_1 = np.array([0., 0.])
        self.offset_corner_2 = np.array([0., 0.])
        self.offset_corner_3 = np.array([0., 0.])
        self.offset_corner_4 = np.array([0., 0.])
        self.calculate_principal_axis()

        self.feature_detector = feature_detector
        self.initial_tracking_feature_points = cv.goodFeaturesToTrack(frame_gray, mask=self.mask_tracking_space,
                                                                      **FEATURE_PARAMS)

        if self.initial_tracking_feature_points is not None:
            self.tracking_feature_points = self.initial_tracking_feature_points
            self.gray = frame_gray
            self.color = frame_color
        else:
            logging.warning("static_object_tracker: %s has no initial tracking feature points "
                  "will unsubscribe now" % self.name)
            self.unregister("tracking has no initial tracking feature points!")
            return

        self.traj = [center]

        self.prev_center = center
        self.initial_center = center
        self.ave_center = center

        self.bounding_polygon = None
        self.update_bounding_polygon()

         # Note: this is the minimum bounding box, not the Axis-aligned-bounding-box
        self.bbox = [(polygon_length, polygon_width)]
        self.prev_bbox = (polygon_length, polygon_width)
        self.bbox_aabb = [(bbox_width, bbox_height)]
        self.prev_bbox_aabb = (bbox_width, bbox_height)

        self.initial_length = polygon_length
        self.initial_width = polygon_width
        self.initial_aabb_width = bbox_width
        self.initial_aabb_height = bbox_height
        
        self.max_length = polygon_length
        self.max_width = polygon_width
        self.ave_length = polygon_length
        self.ave_width = polygon_width
        self.ave_speed_width = 0
        self.ave_speed_height = 0

        self.last_detected_frame_id = -1
        self.width = frame_gray.shape[1]
        self.height = frame_gray.shape[0]

        self.time_in_deadzone = 0
        self.prev_A = None

        self.position_updated_by_detection = True

        self.list_polygons_occlusion_zone = None
        self.is_recongized_by_occulusion_zone_shift = False
        self.num_large_jump = 0
        self.speeds_width = [0]
        self.speeds_height = [0]
        self.longitudinal_directions = [np.array([1, 0]).astype(np.float)]
        self.ave_longitudinal_direction = np.array([1, 0]).astype(np.float)
        self.init_plotting_parameters()
        self.offset_w_name = -50
        self.init_grid(num_row, num_col)
        self.size_circle_skeleton = 8
        self.size_line_skeleton = 4

        self.is_dynamic = False

        for other_static_agent in StaticObjectTracker.global_static_agents:
            if not (other_static_agent.index_row == self.index_row and other_static_agent.index_col == self.index_col):
                continue
            
            ave_norm = other_static_agent.smaller_principal_axis_vector + self.smaller_principal_axis_vector
            ave_norm = ave_norm / np.linalg.norm(ave_norm)
            dist = np.abs(np.dot((other_static_agent.ave_center - self.ave_center), ave_norm))
            StaticObjectTracker.global_static_lane_width_min = min(StaticObjectTracker.global_static_lane_width_min,
                                                                   dist)

            min_dist = StaticObjectTracker.global_static_lane_width[self.index_grid_cell]

            if min_dist is None:
                StaticObjectTracker.global_static_lane_width[self.index_grid_cell] = dist
            else:
                StaticObjectTracker.global_static_lane_width[self.index_grid_cell] = min(min_dist, dist)

        StaticObjectTracker.global_static_agents.append(self)

        self.num_uses_to_build_lane = 0

        self.set_last_seen_frame_id(frame_counter)
        self.index_pose_tracker = None
        self.is_initialized = True

    def init_grid(self, num_row, num_col):
        self.num_row = num_row
        self.num_col = num_col
        self.width_interval = self.width / self.num_col
        self.height_interval = self.height / self.num_row
        self.index_col = int(self.ave_center[0] / self.width_interval)
        self.index_row = int(self.ave_center[1] / self.height_interval)
        self.index_grid_cell = self.index_row * self.num_col + self.index_col
        if StaticObjectTracker.global_static_lane_width is None:
            StaticObjectTracker.global_static_lane_width = [None for _ in range(num_row * num_col)]

    def plot_skeleton(self, larger_principal_axis_point_1, ratio_x, ratio_y, frame, color_code, rescaled_center):
        xii, yii = larger_principal_axis_point_1
        xii, yii = int(xii * ratio_x), int(yii * ratio_y)
        rescaled_point = np.array([xii, yii]).astype(np.int32)

        cv.circle(frame, tuple(rescaled_point), self.size_circle_skeleton, color_code, -1)
        cv.line(frame, tuple(rescaled_center), tuple(rescaled_point), color_code, self.size_line_skeleton)

    def plot_traj(self, frame, ratio_x, ratio_y, is_plot_polygon_requested, is_plot_car_heading_requested,
                  is_plot_feature_points_requested, size_circle, line_width, agent_id_text_size, agent_id_font_size,
                  agent_id_offset_w, agent_id_offset_h):
        len_traj = len(self.traj)
        if len_traj < 1:
            logging.error("Something wrong with static object %s, requested to plot traj, "
                  "but len of traj smaller than 1" % self.obj_id)
            return frame

        xi, yi = self.ave_center

        larger_principal_axis_point_1 = self.larger_principal_axis_point_1 + self.ave_center
        larger_principal_axis_point_2 = self.larger_principal_axis_point_2 + self.ave_center

        smaller_principal_axis_point_1 = self.smaller_principal_axis_point_1 + self.ave_center
        smaller_principal_axis_point_2 = self.smaller_principal_axis_point_2 + self.ave_center

        xi, yi = int(xi * ratio_x), int(yi * ratio_y)
        rescaled_center = np.array([xi, yi]).astype(np.int32)

        cv.circle(frame, tuple(rescaled_center), self.size_circle, self.color_code, -1)

        self.plot_skeleton(larger_principal_axis_point_1, ratio_x, ratio_y, frame,
                           self.color_code_larger_principal_axis, rescaled_center)
        self.plot_skeleton(larger_principal_axis_point_2, ratio_x, ratio_y, frame,
                           self.color_code_larger_principal_axis, rescaled_center)
        self.plot_skeleton(smaller_principal_axis_point_1, ratio_x, ratio_y, frame,
                           self.color_code_smaller_principal_axis, rescaled_center)
        self.plot_skeleton(smaller_principal_axis_point_2, ratio_x, ratio_y, frame,
                           self.color_code_smaller_principal_axis, rescaled_center)

        frame = cv.putText(frame, self.display_name, (xi + self.offset_w_name, yi + self.offset_h_name),
                           cv.FONT_HERSHEY_SIMPLEX, self.font_size_name, self.color_code, 2, cv.LINE_AA)

        return frame

    def is_feature_update_qualified(self, box_center, bbox_width, bbox_height, bounding_polygon, polygon_length,
                                    polygon_width, frame_counter, index_detection=None):
        is_overlap = self.is_duplicate(box_center, bbox_width, bbox_height)
        return is_overlap, is_overlap

    def is_duplicate(self, box_center, bbox_width, bbox_height, index_detection=None):
        """
        scaled_mask is supposed to be re-scaled from detection space dimension to tracking space dimension already
        """
        dist = self.bounding_polygon.distance(geometry.Point(box_center))
        return dist < 25 * 0.5

    def update_feature(self, agent_class, mask_detection, mask_tracking, center, polygon_length, polygon_width,
                       frame_counter):

        self.set_last_seen_frame_id(frame_counter)

        if self.index_to_update == 0:
            x_n_minus_1, y_n_minus_1 = self.prev_center
            x_n, y_n = center

            self.prev_center = center
            self.traj.append(center)

            self.prev_bbox = (polygon_length, polygon_width)
            self.max_length = max(self.max_length, polygon_length)
            self.max_width = max(self.max_width, polygon_width)
            self.bbox.append(self.prev_bbox)

            self.update_dynamics(x_n, y_n, x_n_minus_1, y_n_minus_1)

            self.mask_detection_space = mask_detection
            self.mask_tracking_space = mask_tracking

            self.calculate_principal_axis()

            self.update_bounding_polygon()

        self.index_to_update = (self.index_to_update + 1) % self.period_to_update
        self.position_updated_by_detection = True
        self.num_updates_from_detection += 1

    def update_dynamics(self, x_n, y_n, x_n_minus_1, y_n_minus_1):
        self.ave_center = self.get_mva(MOVING_AVERAGE_WINDOW_START, MOVING_AVERAGE_WINDOW_SIZE, self.traj)
        self.ave_length, self.ave_width = self.get_mva_bbox(MOVING_AVERAGE_WINDOW_START, MOVING_AVERAGE_WINDOW_SIZE,
                                                            self.bbox)
        self.index_col = int(self.ave_center[0] / self.width_interval)
        self.index_row = int(self.ave_center[1] / self.height_interval)
        self.index_grid_cell = self.index_row * self.num_col + self.index_col

    def calculate_principal_axis(self):
        try:
            start_time = timeit.default_timer()

            y, x = np.nonzero(self.mask_detection_space)
            x = x - np.mean(x)
            y = y - np.mean(y)
            coords = np.vstack([x, y])
            cov = np.cov(coords)

            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]

            principal_axis_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue,
            # eigenvalues from large to small
            principal_axis_v2 = evecs[:, sort_indices[1]]

            x_v1, y_v1 = principal_axis_v1
            x_v2, y_v2 = principal_axis_v2

            # transform from detection space to tracking space
            x_v1 *= self.scale_width_det_to_tracking
            x_v2 *= self.scale_width_det_to_tracking
            y_v1 *= self.scale_height_det_to_tracking
            y_v2 *= self.scale_height_det_to_tracking

            principal_axis_v1 = np.array([x_v1, y_v1])
            principal_axis_v2 = np.array([x_v2, y_v2])
            principal_axis_v1 /= np.linalg.norm(principal_axis_v1)
            principal_axis_v2 /= np.linalg.norm(principal_axis_v2)

            x_v1, y_v1 = principal_axis_v1
            x_v2, y_v2 = principal_axis_v2

            index_x_max = np.argmax(x)
            index_x_min = np.argmin(x)
            index_y_max = np.argmax(y)
            index_y_min = np.argmin(y)

            x_max_point_proj_larger_axis, x_max_point_proj_smaller_axis = \
                compute_eigenaxis_points(index_x_max, x, y, principal_axis_v1, principal_axis_v2,
                                         self.scale_width_det_to_tracking, self.scale_height_det_to_tracking)

            x_min_point_proj_larger_axis, x_min_point_proj_smaller_axis = \
                compute_eigenaxis_points(index_x_min, x, y, principal_axis_v1, principal_axis_v2,
                                         self.scale_width_det_to_tracking, self.scale_height_det_to_tracking)

            y_max_point_proj_larger_axis, y_max_point_proj_smaller_axis = \
                compute_eigenaxis_points(index_y_max, x, y, principal_axis_v1, principal_axis_v2,
                                         self.scale_width_det_to_tracking, self.scale_height_det_to_tracking)

            y_min_point_proj_larger_axis, y_min_point_proj_smaller_axis = \
                compute_eigenaxis_points(index_y_min, x, y, principal_axis_v1, principal_axis_v2,
                                         self.scale_width_det_to_tracking, self.scale_height_det_to_tracking)

            larger_axis_max = max(x_max_point_proj_larger_axis, x_min_point_proj_larger_axis,
                                  y_max_point_proj_larger_axis, y_min_point_proj_larger_axis)
            larger_axis_min = min(x_max_point_proj_larger_axis, x_min_point_proj_larger_axis,
                                  y_max_point_proj_larger_axis, y_min_point_proj_larger_axis)

            larger_range = (larger_axis_max - larger_axis_min) * 0.5

            smaller_axis_max = max(x_max_point_proj_smaller_axis, x_min_point_proj_smaller_axis,
                                y_max_point_proj_smaller_axis, y_min_point_proj_smaller_axis)

            smaller_axis_min = min(x_max_point_proj_smaller_axis, x_min_point_proj_smaller_axis,
                                   y_max_point_proj_smaller_axis, y_min_point_proj_smaller_axis)

            smaller_range = (smaller_axis_max - smaller_axis_min) * 0.5

            smaller_range = StaticObjectTracker.global_static_lane_marker_width = min(smaller_range,
                                                                      StaticObjectTracker.global_static_lane_marker_width)
            StaticObjectTracker.global_static_lane_marker_width = smaller_range

            self.larger_principal_axis_point_1 = np.array([x_v1 * -larger_range, y_v1 * -larger_range])
            self.larger_principal_axis_point_2 = np.array([x_v1 * larger_range, y_v1 * larger_range])

            self.smaller_principal_axis_point_1 = np.array([x_v2 * -smaller_range, y_v2 * -smaller_range])
            self.smaller_principal_axis_point_2 = np.array([x_v2 * smaller_range, y_v2 * smaller_range])

            self.larger_principal_axis_vector = self.larger_principal_axis_point_2 - self.larger_principal_axis_point_1
            self.larger_principal_axis_vector = self.larger_principal_axis_vector /\
                np.linalg.norm(self.larger_principal_axis_vector)

            self.smaller_principal_axis_vector = self.smaller_principal_axis_point_2 - self.smaller_principal_axis_point_1
            self.smaller_principal_axis_vector = self.smaller_principal_axis_vector /\
                np.linalg.norm(self.smaller_principal_axis_vector)

            self.offset_corner_1 = self.larger_principal_axis_point_1 + self.smaller_principal_axis_point_1
            self.offset_corner_2 = self.larger_principal_axis_point_2 + self.smaller_principal_axis_point_1
            self.offset_corner_3 = self.larger_principal_axis_point_2 + self.smaller_principal_axis_point_2
            self.offset_corner_4 = self.larger_principal_axis_point_1 + self.smaller_principal_axis_point_2

            ObjectTracker.global_num_calculate_principal_axis += 1
            ObjectTracker.global_time_calculate_principal_axis += timeit.default_timer() - start_time

        except Exception as e:
            logging.warning("static_object_tracker: %s" % str(e))

