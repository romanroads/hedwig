import cv2 as cv
import numpy as np
from shapely import geometry
import timeit
import logging

from constant_values import *


class ObjectTracker:
    global_sum_detection_efficiency = 0
    global_count_detection_efficiency = 0

    global_num_calculate_principal_axis = 0
    global_time_calculate_principal_axis = 0

    global_num_good_feature_to_track = 0
    global_time_good_feature_to_track = 0.

    global_num_orb = 0
    global_time_orb = 0.

    global_num_calculate_optical_flow = 0
    global_time_calculate_optical_flow = 0.

    global_num_dnn_feature_points = 0
    global_time_dnn_feature_points = 0.

    global_num_estimate_affine = 0
    global_time_estimate_affine = 0.

    global_list_tracklet_id = []
    global_list_tracklet_length = []
    global_list_tracklet_width = []

    global_dict_distance = {}
    global_dict_time = {}
    global_dict_first_frame = {}
    global_dict_last_frame = {}
    global_dict_front_agent_id = {}
    global_dict_front_agent_frac_dist = {}
    global_dict_rear_agent_id = {}
    global_dict_rear_agent_frac_dist = {}

    global_dict_road_lane_id_to_f_slot = {}
    global_dict_road_lane_id_to_agent_id = {}

    global_total_registered_agents = 0
    global_total_agents_not_reach_unsub_zone = 0

    def __init__(self, obj_id, list_occ, frame_gray, mask_detection, mask_tracking, center, polygon_length, polygon_width,
                 bbox_width, bbox_height,
                 fps,
                 av_list, agent_class, num_row, num_col, larger_principal_axis_point_1, larger_principal_axis_point_2,
                 smaller_principal_axis_point_1, smaller_principal_axis_point_2, larger_principal_axis_vector,
                 smaller_principal_axis_vector, offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4,
                 frame_counter, array_feature_point, feature_detector="gftt", frame_color=None):
        """
        here frame can NOT have authentic resolution
        but frame_gray is having the authentic resolution
        :param obj_id:
        :param frame:
        :param frame_gray:
        :param rmask:
        :param center:
        """
        self.is_initialized = False
        self.obj_id = obj_id
        self.duration_lost = -1
        self.agent_class = agent_class
        self.color_code = COLOR_MATRIX[self.obj_id % 100].tolist()
        self.thickness = 4

        self.mask_detection_space = mask_detection
        self.mask_tracking_space = mask_tracking
        self.scale_width_det_to_tracking = mask_tracking.shape[1] / mask_detection.shape[1]
        self.scale_height_det_to_tracking = mask_tracking.shape[0] / mask_detection.shape[0]

        self.larger_principal_axis_point_1, self.larger_principal_axis_point_2, self.smaller_principal_axis_point_1,\
            self.smaller_principal_axis_point_2, self.larger_principal_axis_vector, self.smaller_principal_axis_vector,\
            self.offset_corner_1, self.offset_corner_2, self.offset_corner_3, self.offset_corner_4 = \
            larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1, \
            smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector, \
            offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4

        if self.obj_id in av_list:
            self.name = "AV%d" % self.obj_id
            self.display_name = "%d" % self.obj_id
        else:
            three_letter_category_name = self.agent_class[0:3]
            self.name = "%s%d" % (self.agent_class, self.obj_id)
            self.display_name = "%d" % self.obj_id if self.is_a_sedan() \
                else "%s%d" % (three_letter_category_name, self.obj_id)

        self.fps = fps

        # Note: if agent has no detection update from DNN for too long, we unsubscribe this agent
        self.staled_detection_threshold = TIME_THRESHOLD_TO_DEFINE_TRACKLET_LOST[self.agent_class]

        # Note agent has to be alive for much longer than the staled detection threshold...to be called "missed"
        self.thresh_min_live_steps = self.staled_detection_threshold * self.fps * 5.

        self.frame_counter = frame_counter

        # Note: this counter will increment whenever detection DNN feeds in a measurement or update
        self.num_updates_from_detection = 0

        # Note: this counter will increment at every frame while the agent is registered in our system
        self.num_updates_from_tracking = 0

        self.feature_detector = feature_detector

        self.tracking_feature_points = None
        self.prev_tracking_feature_points = None

        if frame_gray is not None:
            self.width = frame_gray.shape[1]
            self.height = frame_gray.shape[0]

            start_time = timeit.default_timer()

            if self.feature_detector == "orb":
                self.orb = cv.ORB_create(nfeatures=100, scoreType=cv.ORB_FAST_SCORE)
                kps, _ = self.orb.detectAndCompute(frame_color, mask=self.mask_tracking_space)

                feature_points = np.array([[[kp.pt[0], kp.pt[1]]] for kp in kps]).astype(np.float32)

                if len(feature_points) <= 0:
                    feature_points = cv.goodFeaturesToTrack(frame_gray, mask=self.mask_tracking_space, **FEATURE_PARAMS)

                self.initial_tracking_feature_points = feature_points
                ObjectTracker.global_time_orb += timeit.default_timer() - start_time
                ObjectTracker.global_num_orb += 1

            elif self.feature_detector == "gftt":
                self.initial_tracking_feature_points = cv.goodFeaturesToTrack(frame_gray, mask=self.mask_tracking_space,
                                                                              **FEATURE_PARAMS)

                ObjectTracker.global_time_good_feature_to_track += timeit.default_timer() - start_time
                ObjectTracker.global_num_good_feature_to_track += 1

            elif self.feature_detector == "none":
                self.initial_tracking_feature_points = array_feature_point[:, :2].reshape(3, 1, 2).astype(np.float32)
                ObjectTracker.global_time_dnn_feature_points += timeit.default_timer() - start_time
                ObjectTracker.global_num_dnn_feature_points += 1

            if self.initial_tracking_feature_points is not None:
                self.tracking_feature_points = self.initial_tracking_feature_points
                self.prev_tracking_feature_points = self.initial_tracking_feature_points
                self.gray = frame_gray
                self.color = frame_color
            else:
                logging.warning("object_tracker: %s has no initial tracking feature points "
                      "will unsubscribe now" % self.name)
                self.unregister("tracking has no initial tracking feature points!")
                return
        else:
            self.width = mask_tracking.shape[1]
            self.height = mask_tracking.shape[0]
            self.gray = None
            self.color = None

        self.traj = [center]
        self.traj_filtered = []

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

        self.time_in_deadzone = 0
        self.prev_A = None

        self.position_updated_by_detection = True
        self.list_polygons_occlusion_zone = list_occ
        self.is_recongized_by_occulusion_zone_shift = False
        self.num_large_jump = 0
        self.speeds_width = [0]
        self.speeds_height = [0]
        self.longitudinal_directions = [np.array([1, 0]).astype(np.float)]
        self.ave_longitudinal_direction = np.array([1, 0]).astype(np.float)

        self.thresh_frac_length, self.thresh_frac_width = 1., 1.
        self.init_plotting_parameters()

        self.init_grid(num_row, num_col)
        self.is_dynamic = True

        self.set_last_seen_frame_id(frame_counter)

        self.array_feature_points = array_feature_point
        self.inferred_array_feature_points = array_feature_point

        if self.array_feature_points is not None:
            thresh_precisions = FEATURE_POINT_MAPS[self.agent_class]['precisions']

            self.is_array_feature_points_confident = self.array_feature_points[0, 2] > thresh_precisions[0] and \
                self.array_feature_points[1, 2] > thresh_precisions[1] and \
                self.array_feature_points[2, 2] > thresh_precisions[2]
        else:
            self.is_array_feature_points_confident = False

        self.index_pose_tracker = None
        self.is_initialized = True
        self.is_in_occlusion_zone = False
        self.safe_tracking_scale_factor = 2.

        # Note: threshold to lock y axis movement when in occ zone, value of zero being very strict, no y movement
        # self.thresh_occ = 3.
        self.thresh_occ = 0.

    def is_a_sedan(self):
        return self.agent_class is "Sedan"

    def is_a_truck(self):
        return self.agent_class is "Truck"

    def init_grid(self, num_row, num_col):
        self.num_row = num_row
        self.num_col = num_col
        self.width_interval = self.width / self.num_col
        self.height_interval = self.height / self.num_row
        self.index_col = int(self.ave_center[0] / self.width_interval)
        self.index_row = int(self.ave_center[1] / self.height_interval)
        self.index_grid_cell = self.index_row * self.num_col + self.index_col

    def init_plotting_parameters(self):
        self.size_circle = 5 if self.agent_class == "Pedestrian" else 16
        self.offset_w_name = -60 if self.agent_class == "Pedestrian" else -40
        self.offset_h_name = 12 if self.agent_class == "Pedestrian" else 12
        self.font_size_name = 0.5 if self.agent_class == "Pedestrian" else 1
        self.traj_len_display = 100 if self.agent_class == "Pedestrian" else 50

        self.thresh_frac_length = 10.0 if self.is_a_truck() else 1.5
        self.thresh_frac_width = 10.0 if self.is_a_truck() else 1.5

        self.is_dimension_changed_a_lot_due_to_turning = False

    @staticmethod
    def propogate_feature_points(prev_img, next_img, prev_feature_points):
        start_time = timeit.default_timer()

        feature_points, status, error = cv.calcOpticalFlowPyrLK(prev_img, next_img, prev_feature_points, None,
                                                                **LK_PARAMS)

        ObjectTracker.global_time_calculate_optical_flow += timeit.default_timer() - start_time
        ObjectTracker.global_num_calculate_optical_flow += 1

        return feature_points, status, error

    def track(self, new_gray, use_homography, use_ransac, frame_counter, new_color=None):
        """
        The current timing goes like detection first, tracking next, detection happens less often than
        tracking, tracking should happen every frame. Therefore if detection updated the center and bbox, we give
        up updating them again in tracking, higher priority for detection or DNN
        """

        self.num_updates_from_tracking += 1

        # Note: since track method runs every frame, if for this frame DNN has feed in its measurement, we stop
        # using Optical Flow to track, 100% rely on DNN for this frame
        if self.position_updated_by_detection:
            self.gray = new_gray
            self.color = new_color
            self.position_updated_by_detection = False
            return True

        cur_feature = self.tracking_feature_points

        self.frame_counter = frame_counter

        # Note this sets a baseline for how fast this car is travelling
        v_x_mean = np.mean(self.speeds_width)
        v_y_mean = np.mean(self.speeds_height)

        # Note: this follows similar shape as Open CV affine matrix 2 x 3
        translation_affine = np.array([[1, 0, v_x_mean],
                                       [0, 1, v_y_mean]]).astype(np.float32)

        x_n_minus_1, y_n_minus_1 = self.prev_center

        x_corner_1, y_corner_1 = int(x_n_minus_1 + self.offset_corner_1[0]), int(
            y_n_minus_1 + self.offset_corner_1[1])
        x_corner_2, y_corner_2 = int(x_n_minus_1 + self.offset_corner_2[0]), int(
            y_n_minus_1 + self.offset_corner_2[1])
        x_corner_3, y_corner_3 = int(x_n_minus_1 + self.offset_corner_3[0]), int(
            y_n_minus_1 + self.offset_corner_3[1])
        x_corner_4, y_corner_4 = int(x_n_minus_1 + self.offset_corner_4[0]), int(
            y_n_minus_1 + self.offset_corner_4[1])

        # Note: basic feature points are the 5 points: center point plus the AABB 4 corner points
        basic_points = [(x_n_minus_1, y_n_minus_1), (x_corner_1, y_corner_1), (x_corner_2, y_corner_2),
                        (x_corner_3, y_corner_3), (x_corner_4, y_corner_4)]

        basic_points_arr = np.array(basic_points).astype(np.float32)
        n_basic = basic_points_arr.shape[0]
        parity = np.ones(n_basic).reshape(n_basic, 1).astype(np.float32)

        # Note: N x 3 matrix
        basic_points_arr_parity = np.concatenate((basic_points_arr, parity), axis=1)

        # Note: matrix multiplication [N x 3] x [3 x 2] = [N x 2]
        basic_points_arr_next = np.matmul(basic_points_arr_parity, translation_affine.T)

        basic_points_next = list(basic_points_arr_next)

        self.is_in_occlusion_zone = False
        if self.list_polygons_occlusion_zone is not None:
            if len(self.list_polygons_occlusion_zone) > 0:
                for poly in self.list_polygons_occlusion_zone:
                    if poly.contains(geometry.Point(x_n_minus_1, y_n_minus_1)):
                        self.is_in_occlusion_zone = True
                        break

        if self.is_in_occlusion_zone:
            filtered_feature_points = basic_points
            filtered_feature_points_next = basic_points_next
        else:

            filtered_feature_points = []
            filtered_feature_points_next = []

            if self.feature_detector == "gftt":
                next_tracking_feature_points, trace_status, track_err = self.propogate_feature_points(self.gray, new_gray,
                                                                                cur_feature)

            elif self.feature_detector == "orb":
                next_tracking_feature_points, trace_status, track_err = self.propogate_feature_points(self.color, new_color,
                                                                                           cur_feature)

            elif self.feature_detector == "none":
                next_tracking_feature_points, trace_status, track_err = self.propogate_feature_points(self.gray, new_gray,
                                                                                           cur_feature)

            # Note: safe region for the optical flow feature points group
            n_feature_points = cur_feature.shape[0]
            for i_p in range(n_feature_points):
                prev_p = cur_feature[i_p]
                next_p = next_tracking_feature_points[i_p]
                prev_p_x, prev_p_y = prev_p[0][0], prev_p[0][1]
                next_p_x, next_p_y = next_p[0][0], next_p[0][1]
                prev_p_tuple = (prev_p_x, prev_p_y)
                next_p_tuple = (next_p_x, next_p_y)
                prev_point = geometry.Point(prev_p_tuple)
                dist = self.bounding_polygon.distance(prev_point)

                if dist <= 0:
                    filtered_feature_points.append(prev_p_tuple)
                    filtered_feature_points_next.append(next_p_tuple)

            if len(filtered_feature_points) <= 0:
                filtered_feature_points.extend(basic_points)
                filtered_feature_points_next.extend(basic_points_next)

        assert len(filtered_feature_points) == len(filtered_feature_points_next),\
            "object_tracker: feature points dimen do not match!"

        assert len(filtered_feature_points) > 0, "object_tracker: feature points dimen is zero!"

        n_filtered_points = len(filtered_feature_points)

        next_tracking_feature_points = np.array(filtered_feature_points_next).\
            reshape(n_filtered_points, 1, 2).astype(np.float32)

        cur_feature = np.array(filtered_feature_points).reshape(n_filtered_points, 1, 2).astype(np.float32)

        time_start = timeit.default_timer()

        # Note: this affine transformation matrix is 2 x 3, with the 3rd matrix element being parity element
        # cur feature points and next feature points have shape of N x 1 x 2, 2nd dimen is the dummy dimension
        affine, status = cv.estimateAffinePartial2D(cur_feature, next_tracking_feature_points,
                                                    method=cv.RANSAC, ransacReprojThreshold=0.1)

        ObjectTracker.global_time_estimate_affine += timeit.default_timer() - time_start
        ObjectTracker.global_num_estimate_affine += 1

        if affine is None:
            affine = self.prev_A

        if affine is None:
            logging.warning("object_tracker: %s has no Affine and also no previous affine, "
                  "will unsubscribe now" % self.obj_id)
            self.unregister("tracking has no previous and current affine!")
            return False

        # Note handle the affine transform calculation failures
        # safe region for the affine transformation
        affine_computation_success = 1.
        n_affine_points = len(status)
        if n_affine_points > 0:
            for i in range(n_affine_points):
                next_p = next_tracking_feature_points[i]
                next_p_x, next_p_y = next_p[0][0], next_p[0][1]
                next_p_tuple = (next_p_x, next_p_y)
                next_p_point = geometry.Point(next_p_tuple)
                dist = self.bounding_polygon.distance(next_p_point)

                if dist > np.abs(v_x_mean) * self.safe_tracking_scale_factor:
                    affine_computation_success = 0.
                    break

        if affine_computation_success <= 0:
            affine = translation_affine
            cur_feature_squeezed = np.squeeze(cur_feature, axis=1)
            n_rows = cur_feature_squeezed.shape[0]
            parity = np.ones(n_rows).reshape(n_rows, 1).astype(np.float32)
            cur_feature_squeezed = np.concatenate((cur_feature_squeezed, parity), axis=1)
            next_feature = np.matmul(cur_feature_squeezed, affine.T)
            next_tracking_feature_points = np.expand_dims(next_feature, axis=1)

        self.prev_A = affine

        center = np.matmul(np.append(self.prev_center, 1), affine.T)

        x_n, y_n = center

        # TODO should be able to get rid of this hard cut.... since when in occ zone, the optical flow feature points
        # are generated by the translation transform, which should guarantee a zero vertical affine from CV computation
        if self.is_in_occlusion_zone:
            v_y_mean_amplitude = np.abs(v_y_mean)

            # Note: [N x 3] x [3 x 2] = N x 2
            center_simulated = np.matmul(np.append(self.prev_center, 1), translation_affine.T)
            dist_y = np.abs(center_simulated[1] - center[1])

            if self.obj_id == 32:
                print("joe track ....", center_simulated, center, dist_y, "<", v_y_mean_amplitude)

            # Note: if using historical average of speed to move the car and the simulated position and DNN position
            # are very different, we do not allow vertical jump movements...
            if dist_y >= v_y_mean_amplitude * self.thresh_occ:
                y_n = y_n_minus_1
                center = (x_n, y_n)

            if self.obj_id == 32:
                print("joe track .... final", self.frame_counter, center)

        self.update_dynamics(x_n, y_n, x_n_minus_1, y_n_minus_1)

        self.traj.append(center)

        self.prev_center = center
        self.update_bounding_polygon()

        self.bbox.append(self.prev_bbox)
        self.bbox_aabb.append(self.prev_bbox_aabb)

        self.gray = new_gray
        self.color = new_color

        self.prev_tracking_feature_points = cur_feature
        self.tracking_feature_points = next_tracking_feature_points

        return True

    def track_combined(self, x_center, y_center, frame_counter):
        """
        The current timing goes like detection first, tracking next, detection happens less often than
        tracking, tracking should happen every frame. Therefore if detection updated the center and bbox, we give
        up updating them again in tracking, higher priority for detection or DNN
        """
        self.frame_counter = frame_counter

        x_n_minus_1, y_n_minus_1 = self.prev_center

        is_in_occlusion_zone = False
        if self.list_polygons_occlusion_zone is not None:
            if len(self.list_polygons_occlusion_zone) > 0:
                for poly in self.list_polygons_occlusion_zone:
                    if poly.contains(geometry.Point(x_n_minus_1, y_n_minus_1)):
                        is_in_occlusion_zone = True
                        break

        delta_dist = np.sqrt((x_n_minus_1 - x_center) ** 2 + (y_n_minus_1 - y_center) ** 2)
        delta_y = y_center - y_n_minus_1
        delta_x = x_center - x_n_minus_1

        x_n, y_n = x_center, y_center
        center = np.array([x_center, y_center])

        self.update_dynamics(x_n, y_n, x_n_minus_1, y_n_minus_1)

        self.traj.append(center)
        self.prev_center = center

        self.update_bounding_polygon()

        self.bbox.append(self.prev_bbox)
        self.bbox_aabb.append(self.prev_bbox_aabb)

        return True

    def update_dynamics(self, x_n, y_n, x_n_minus_1, y_n_minus_1):
        delta_width = x_n - x_n_minus_1
        delta_height = y_n - y_n_minus_1

        self.speeds_width.append(delta_width)
        self.speeds_height.append(delta_height)

        self.ave_speed_width = self.get_mva(MOVING_AVERAGE_WINDOW_START,
                                            MOVING_AVERAGE_WINDOW_SIZE,
                                            self.speeds_width)
        self.ave_speed_height = self.get_mva(MOVING_AVERAGE_WINDOW_START,
                                             MOVING_AVERAGE_WINDOW_SIZE,
                                             self.speeds_height)

        self.ave_longitudinal_direction = self.get_mva(MOVING_AVERAGE_WINDOW_START,
                                             MOVING_AVERAGE_WINDOW_SIZE,
                                            self.longitudinal_directions)

        self.ave_center = self.get_mva(MOVING_AVERAGE_WINDOW_START, MOVING_AVERAGE_WINDOW_SIZE, self.traj)
        self.traj_filtered.append(self.ave_center)

        # Note here the update has to been immediate, not the moving average which will be having latency
        self.index_col = np.clip(int(x_n / self.width_interval), 0, self.num_col - 1)
        self.index_row = np.clip(int(y_n / self.height_interval), 0, self.num_row - 1)
        self.index_grid_cell = self.index_row * self.num_col + self.index_col

    @staticmethod
    def compute_physics_constraint(speed_mva, speed, radius):
        if np.abs(speed_mva) < 1:
            return np.abs(speed) > radius * 0.5
        else:
            return np.abs(speed / speed_mva) > MAX_TRACKING_DIST_JUMP_SCALE

    def plot_tracking_feature_poins(self, frame, ratio_x_track_to_output, ratio_y_track_to_output, size_circle):
        # Note: the red dots are the previous state of the tracking feature points while the yellow ones are the
        # current state
        n_points = self.prev_tracking_feature_points.shape[0]
        for i in range(n_points):
            point = self.prev_tracking_feature_points[i]
            x, y = point[0][0], point[0][1]
            x, y = int(x * ratio_x_track_to_output), int(y * ratio_y_track_to_output)
            # Note to prevent super large pixel coordinate plotting
            try:
                cv.circle(frame, (x, y), size_circle, RED, -1)
            except:
                return

        n_points = self.tracking_feature_points.shape[0]
        for i in range(n_points):
            point = self.tracking_feature_points[i]
            x, y = point[0][0], point[0][1]
            x, y = int(x * ratio_x_track_to_output), int(y * ratio_y_track_to_output)
            # Note to prevent super large pixel coordinate plotting
            try:
                cv.circle(frame, (x, y), size_circle, YELLOW_RGB, -1)
            except:
                return

    def plot_traj(self, frame, ratio_x, ratio_y, is_plot_polygon_requested, is_plot_car_heading_requested,
                  is_plot_feature_points_requested, size_circle, line_width, agent_id_text_size, agent_id_font_size,
                  agent_id_offset_w, agent_id_offset_h):

        len_traj = len(self.traj_filtered)

        xi, yi = self.prev_center

        xi, yi = int(xi * ratio_x), int(yi * ratio_y)

        # Note to prevent super large pixel coordinate plotting
        try:
            cv.circle(frame, (xi, yi), size_circle, self.color_code, -1)
        except:
            return

        i = len_traj - 1
        counter = 0
        xi_0 = xi
        yi_0 = yi
        interval = 10
        duty_cycle = 0.6
        while i > 0 and counter < self.traj_len_display:
            xi_1, yi_1 = self.traj_filtered[i]
            xi_1, yi_1 = int(xi_1 * ratio_x), int(yi_1 * ratio_y)

            if "AV" in self.name:
                frac = float(counter % interval) / interval
                if frac < duty_cycle:
                    frame = cv.line(frame, (xi_0, yi_0), (xi_1, yi_1), self.color_code, line_width)
            else:
                frame = cv.line(frame, (xi_0, yi_0), (xi_1, yi_1), self.color_code, line_width)

            xi_0, yi_0 = xi_1, yi_1
            counter += 1
            i -= 1

        # Note: ID of car
        frame = cv.putText(frame, self.display_name, (xi + agent_id_offset_w, yi + agent_id_offset_h),
                           cv.FONT_HERSHEY_SIMPLEX, agent_id_font_size, BLACK, agent_id_text_size,
                           cv.LINE_AA)

        # Note: polygon boundary of the car
        if is_plot_polygon_requested:
            x_corner_1, y_corner_1 = int(xi + self.offset_corner_1[0] * ratio_x), int(
                yi + self.offset_corner_1[1] * ratio_y)
            x_corner_2, y_corner_2 = int(xi + self.offset_corner_2[0] * ratio_x), int(
                yi + self.offset_corner_2[1] * ratio_y)
            x_corner_3, y_corner_3 = int(xi + self.offset_corner_3[0] * ratio_x), int(
                yi + self.offset_corner_3[1] * ratio_y)
            x_corner_4, y_corner_4 = int(xi + self.offset_corner_4[0] * ratio_x), int(
                yi + self.offset_corner_4[1] * ratio_y)

            frame = cv.line(frame, (x_corner_1, y_corner_1), (x_corner_2, y_corner_2), self.color_code, self.thickness)
            frame = cv.line(frame, (x_corner_2, y_corner_2), (x_corner_3, y_corner_3), self.color_code, self.thickness)
            frame = cv.line(frame, (x_corner_3, y_corner_3), (x_corner_4, y_corner_4), self.color_code, self.thickness)
            frame = cv.line(frame, (x_corner_4, y_corner_4), (x_corner_1, y_corner_1), self.color_code, self.thickness)

        # Note: car heading arrow
        if is_plot_car_heading_requested and self.array_feature_points is not None and \
                self.is_array_feature_points_confident:
            end_point = (int(self.array_feature_points[0][0] * ratio_x),
                int(self.array_feature_points[0][1] * ratio_y))
            start_point = (int(self.array_feature_points[2][0] * ratio_x),
                int(self.array_feature_points[2][1] * ratio_y))
            frame = cv.arrowedLine(frame, start_point, end_point,
                            self.color_code, self.thickness, tipLength=0.5)

        if is_plot_feature_points_requested and self.array_feature_points is not None and \
                self.is_array_feature_points_confident:
            for j in range(len(self.array_feature_points)):
                cv.circle(frame, (int(self.array_feature_points[j][0] * ratio_x),
                                  int(self.array_feature_points[j][1] * ratio_y)), 10,
                          FEATURE_POINT_MAPS[self.agent_class]['color_codes'][j]
                          if j < len(FEATURE_POINT_MAPS[self.agent_class]['color_codes']) else (255, 255, 255), -1)

        return frame

    def update_bounding_polygon(self):
        self.bounding_polygon = geometry.Polygon([self.prev_center + self.offset_corner_1,
                                              self.prev_center + self.offset_corner_2,
                                              self.prev_center + self.offset_corner_3,
                                              self.prev_center + self.offset_corner_4])

    def update_feature(self, agent_class, mask_detection, mask_tracking, center, polygon_length, polygon_width,
                       bbox_width, bbox_height,
                       frame_counter, larger_principal_axis_point_1, larger_principal_axis_point_2,
                       smaller_principal_axis_point_1, smaller_principal_axis_point_2, larger_principal_axis_vector,
                       smaller_principal_axis_vector, offset_corner_1, offset_corner_2, offset_corner_3,
                       offset_corner_4, array_feature_points):

        self.set_last_seen_frame_id(frame_counter)

        if self.gray is not None:
            start_time = timeit.default_timer()

            if self.feature_detector == "gftt":
                feature_points = cv.goodFeaturesToTrack(self.gray, mask=mask_tracking, **FEATURE_PARAMS)
                ObjectTracker.global_time_good_feature_to_track += timeit.default_timer() - start_time
                ObjectTracker.global_num_good_feature_to_track += 1

            elif self.feature_detector == "orb":
                kps, _ = self.orb.detectAndCompute(self.color, mask=mask_tracking)
                feature_points = np.array([[[kp.pt[0], kp.pt[1]]] for kp in kps]).astype(np.float32)

                if len(feature_points) <= 0:
                    feature_points = cv.goodFeaturesToTrack(self.gray, mask=mask_tracking, **FEATURE_PARAMS)

                ObjectTracker.global_time_orb += timeit.default_timer() - start_time
                ObjectTracker.global_num_orb += 1

            elif self.feature_detector == "none":
                feature_points = array_feature_points[:, :2].reshape(3, 1, 2).astype(np.float32)
                ObjectTracker.global_time_dnn_feature_points += timeit.default_timer() - start_time
                ObjectTracker.global_num_dnn_feature_points += 1

            if feature_points is not None:
                self.tracking_feature_points = feature_points
                self.prev_tracking_feature_points = self.tracking_feature_points

        x_n_minus_1, y_n_minus_1 = self.prev_center

        self.is_in_occlusion_zone = False
        if self.list_polygons_occlusion_zone is not None:
            if len(self.list_polygons_occlusion_zone) > 0:
                for poly in self.list_polygons_occlusion_zone:
                    if poly.contains(geometry.Point(x_n_minus_1, y_n_minus_1)):
                        self.is_in_occlusion_zone = True
                        break

        x_n, y_n = center

        if self.is_in_occlusion_zone:
            v_x_mean = np.mean(self.speeds_width)
            v_y_mean = np.mean(self.speeds_height)
            v_y_mean_amplitude = np.abs(v_y_mean)
            # Note: this follows similar shape as Open CV affine matrix 2 x 3
            translation_affine_mean = np.array([[1, 0, v_x_mean],
                                                [0, 1, v_y_mean]]).astype(np.float32)

            # Note: [N x 3] x [3 x 2] = N x 2
            center_simulated = np.matmul(np.append(self.prev_center, 1), translation_affine_mean.T)
            dist_y = np.abs(center_simulated[1] - center[1])

            if self.obj_id == 32:
                print("joe", center_simulated, center, dist_y, "<", v_y_mean_amplitude)

            # Note: if using historical average of speed to move the car and the simulated position and DNN position
            # are very different, we do not allow vertical jump movements...
            if dist_y >= v_y_mean_amplitude * self.thresh_occ:
                y_n = y_n_minus_1
                center = (x_n, y_n)

            if self.obj_id == 32:
                print("joe final", self.frame_counter, center)

        delta_width = x_n - x_n_minus_1
        delta_height = y_n - y_n_minus_1

        self.update_dynamics(x_n, y_n, x_n_minus_1, y_n_minus_1)

        self.mask_detection_space = mask_detection
        self.mask_tracking_space = mask_tracking

        self.larger_principal_axis_point_1, self.larger_principal_axis_point_2, self.smaller_principal_axis_point_1,\
            self.smaller_principal_axis_point_2, self.larger_principal_axis_vector, self.smaller_principal_axis_vector,\
            self.offset_corner_1, self.offset_corner_2, self.offset_corner_3, self.offset_corner_4 = \
            larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1, \
            smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector, \
            offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4

        self.traj.append(center)

        self.prev_center = center
        self.update_bounding_polygon()

        self.prev_bbox = (polygon_length, polygon_width)
        self.prev_bbox_aabb = (bbox_width, bbox_height)

        self.max_length = max(self.max_length, polygon_length)
        self.max_width = max(self.max_width, polygon_width)

        self.bbox.append(self.prev_bbox)
        self.bbox_aabb.append(self.prev_bbox_aabb)

        self.ave_length, self.ave_width = self.get_mva_bbox(MOVING_AVERAGE_WINDOW_START, MOVING_AVERAGE_WINDOW_SIZE,
                                                            self.bbox)

        self.array_feature_points = array_feature_points

        thresh_precisions = FEATURE_POINT_MAPS[self.agent_class]['precisions']

        if self.array_feature_points is not None:
            self.is_array_feature_points_confident = self.array_feature_points[0, 2] > thresh_precisions[0] and \
                self.array_feature_points[1, 2] > thresh_precisions[1] and \
                self.array_feature_points[2, 2] > thresh_precisions[2]
        else:
            self.is_array_feature_points_confident = False

        self.agent_class = agent_class
        self.thresh_frac_length = 10.0 if self.is_a_truck() else 1.5
        self.thresh_frac_width = 10.0 if self.is_a_truck() else 1.5

        self.position_updated_by_detection = True
        self.num_updates_from_detection += 1

    def set_last_seen_frame_id(self, f_id):
        self.last_detected_frame_id = f_id

    def is_feature_update_qualified(self, box_center, bbox_width, bbox_height, bounding_polygon, polygon_length,
                                    polygon_width, frame_counter, index_detection=None):
        """
        this function returns two boolean, is the detection matched to a tracklet for update or not, is detection
        overlapped with some existing tracklet (this is for noises, should be removed in the future)
        :param box_center:
        :param bbox_width:
        :param bbox_height:
        :param bounding_polygon:
        :param polygon_length:
        :param polygon_width:
        :param frame_counter:
        :param index_detection:
        :return:
        """
        is_overlap = self.is_duplicate(bounding_polygon, index_detection)

        # Note: if not overlapped, we do not need to consider whether or not to update existing tracklet
        if not is_overlap:
            return False, is_overlap

        # Note: even it is overlapped physically, we need to look at historical data of vehicle length, width to
        # determine whether or not this is a real match for update
        is_matching = self.is_matching_to_historical_detections(polygon_length, polygon_width)

        self.set_last_seen_frame_id(frame_counter)

        return is_matching, is_overlap

    def is_duplicate(self, bounding_polygon, index_detection=None):
        # Note: we tried distance between two polygon, inclusion of two polygons, iou seems the best approach
        # dist_between_two_polygon = self.bounding_polygon.distance(bounding_polygon)
        # is_det_polygon_contain_track_center = bounding_polygon.contains(geometry.Point(self.prev_center))

        intersection = self.bounding_polygon.intersection(bounding_polygon).area
        union = self.bounding_polygon.union(bounding_polygon).area
        iou = intersection / union

        is_iou_too_large = iou > IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP

        return is_iou_too_large

    def is_matching_to_historical_detections(self, polygon_length, polygon_width):

        prev_polygon_length, prev_polygon_width = self.ave_length, self.ave_width
        max_length, min_length = max(prev_polygon_length, polygon_length), min(prev_polygon_length, polygon_length)
        max_width, min_width = max(prev_polygon_width, polygon_width), min(prev_polygon_width, polygon_width)

        ratio_length = max_length / min_length
        ratio_width = max_width / min_width

        f_delta_length = np.abs(ratio_length - 1)
        f_delta_width = np.abs(ratio_width - 1)

        is_size_update_reasonable = f_delta_length < self.thresh_frac_length and f_delta_width < self.thresh_frac_width

        return is_size_update_reasonable

    def not_seen_in_detector(self, cur_frame_id):
        """
        when last-seen frame ID by detection is too much stale, remove it
        :param cur_frame_id:
        :return:
        """
        assert self.last_detected_frame_id >= 0, "[ERROR] object_tracker: something went wrong with last-seen frame id"
        self.duration_lost = (cur_frame_id - self.last_detected_frame_id) / self.fps
        return self.duration_lost > self.staled_detection_threshold

    def is_in_rect(self, xu, yu, xl, yl):
        xn, yn = self.prev_center
        return xu > xn > xl and yu > yn > yl

    def is_in_mask(self, bmask):
        """
        this method needs improvement, when mask is scaled differently, we need to guarantee continuous mask distribution
        :param bmask:
        :return:
        """
        xn, yn = self.prev_center
        xn, yn = int(xn), int(yn)
        return bmask[yn][xn]

    def unregister(self, note):
        if self.num_updates_from_tracking > 0:
            detection_efficiency = self.num_updates_from_detection / self.num_updates_from_tracking * 100
        else:
            detection_efficiency = 0

        ObjectTracker.global_sum_detection_efficiency += detection_efficiency
        ObjectTracker.global_count_detection_efficiency += 1
        ObjectTracker.global_dict_last_frame[self.obj_id] = self.frame_counter

        logging.debug("object_tracker: unregistering tracklet % s due to %s with detection efficiency %.3f %% "
              "duration for the last detection-out-of_sync time %.1f [s]" %
              (self.obj_id, note, detection_efficiency, self.duration_lost))

        # Note: these are the agents that got unsubscribed in a "unusual" way
        if note != TRACKING_LOST_TYPE_UNSUB_ZONE:
            # TODO sometimes agents got quickly registered and unsubscribed at the beginning of their trajectory...
            if self.num_updates_from_tracking > self.thresh_min_live_steps:
                ObjectTracker.global_total_agents_not_reach_unsub_zone += 1
            # TODO when scenario above happens, we deduct it from total number of cars
            else:
                ObjectTracker.global_total_registered_agents -= 1

    def register(self, index_detection=None, index_pose_tracker=None):
        if self.is_dynamic:
            ObjectTracker.global_list_tracklet_id.append(self.obj_id)
            ObjectTracker.global_list_tracklet_length.append(self.initial_length)
            ObjectTracker.global_list_tracklet_width.append(self.initial_width)
            ObjectTracker.global_dict_distance[self.obj_id] = 0.
            ObjectTracker.global_dict_time[self.obj_id] = 0.
            ObjectTracker.global_dict_first_frame[self.obj_id] = self.frame_counter

            ObjectTracker.global_total_registered_agents += 1

        if index_detection is None:
            logging.debug("object_tracker: registering tracklet %s with category %s" % (self.obj_id, self.agent_class))
        else:
            logging.debug("object_tracker: frame %s detection %s is registering tracklet %s with category %s" %
                  (self.frame_counter, index_detection, self.obj_id, self.agent_class))
            self.index_pose_tracker = index_pose_tracker

    @staticmethod
    def get_mva_bbox(window_start_offset, window_size, list_values):
        window_end = min(window_start_offset + window_size, -1)
        ave_w, ave_h, c = 0, 0, 0
        for b in list_values[window_start_offset:window_end]:
            w, h = b
            ave_w += w
            ave_h += h
            c += 1

        if c <= 0:
            return list_values[-1]

        ave_w /= c
        ave_h /= c
        return ave_w, ave_h

    @staticmethod
    def get_mva(window_start_offset, window_size, list_values):
        """
        :param window_start_offset: -10
        :param window_size: 5
        :param list_values:
        :return:
        """
        window_start = window_start_offset
        n_elements = len(list_values)

        if abs(window_start) >= n_elements:
            return list_values[-1]

        window_end = min(window_start_offset + window_size, -1)
        window_len = window_end - window_start

        ave, c = 0., 0

        if len(list_values) > 0:
            if isinstance(list_values[0], tuple):
                ave = np.array([0, 0])

        for w in list_values[window_start:window_end]:
            w = np.array(w)
            ave += w
            c += 1

        if c <= 0:
            return list_values[-1]

        ave = ave / c

        return ave

