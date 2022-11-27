import timeit
from shapely import geometry
import logging
import numpy as np

from object_tracker import ObjectTracker
from static_object_tracker import StaticObjectTracker
from constant_values import *
from pipeline.pipeline import Pipeline


class TrackAgentsIndividualized(Pipeline):

    def __init__(self, num_row, num_col, num_grid_cells, fps, start_id, av_list_str, feature_detector,
                 initial_rot_angle, initial_x, initial_y, is_initial_lane_polygon_adjusted_req,
                 is_to_correct_vehicles, initial_time_window_for_reg):
        # Note: the grid is defined in the detection space, rather the tracking space
        self.num_row = num_row
        self.num_col = num_col
        self.num_grid_cells = num_grid_cells
        self.tracks_distributed_to_grids = [[] for _ in range(self.num_grid_cells)]
        self.fps = fps

        self.cur_id = start_id

        av_list = []
        if len(av_list_str) > 0:
            av_list = [int(i) for i in av_list_str.split(',')]
            logging.debug("TrackAgentsIndividualized: here are the provided list of ID that are AV", av_list)

        self.av_list = av_list

        self.use_ransac = True
        self.use_homography = False

        self.timer_gftt = 0.
        self.num_gftt = 0

        assert feature_detector in ['orb', 'gftt', 'none'],\
            "[ERROR] feature detector type %s not supported" % feature_detector
        self.feature_detector = feature_detector
        logging.debug("TrackAgentsIndividualized: feature detector currently being used: %s" % feature_detector)

        self.is_initial_lane_polygon_adjusted = False
        self.is_initial_lane_polygon_adjusted_req = is_initial_lane_polygon_adjusted_req
        self.initial_rot_angle = initial_rot_angle
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.is_to_correct_vehicles = is_to_correct_vehicles
        self.initial_time_window_for_reg = initial_time_window_for_reg
        super().__init__("TrackAgentsIndividualized")

    def map(self, data):
        start_time = timeit.default_timer()
        self.update_existing_and_create_new_tracklets_using_detection(data)
        self.update_existing_tracklets_using_image_tracking(data)

        # Note: this is the initial rotation and drift to lane polygons
        if self.is_initial_lane_polygon_adjusted_req and not self.is_initial_lane_polygon_adjusted:
            self.initial_lane_polygon_adjustment(data)
            self.is_initial_lane_polygon_adjusted = True

        # Note: this line below will update the lane polygons by drone rotation and drift
        # stablization object will have its correction parameter setup here
        self.check_road_feature_static_tracklets(data)

        # Note: we can use rotational and drift corrections to update vehicle locations
        # here the correction paramsters are already setup above
        if self.is_to_correct_vehicles:
            self.apply_corrections_to_tracklets(data)

        self.compute_relative_lane_slot_measurements(data)

        data[TRACKING_STATISTICS] = (ObjectTracker.global_total_registered_agents,
                                     ObjectTracker.global_total_agents_not_reach_unsub_zone)

        self.timer += timeit.default_timer() - start_time
        return data

    def apply_corrections_to_tracklets(self, data):
        if STABLIZATION_SYSTEM_NAME not in data or data[STABLIZATION_SYSTEM_NAME] is None:
            return

        stablization = data[STABLIZATION_SYSTEM_NAME]
        # Note: x and y shift are pixels in tracking image space
        corr_rot = stablization.correction_rot_angle
        corr_x = stablization.correction_shift_width
        corr_y = stablization.correction_shift_height
        center_point_w, center_point_h = stablization.center

        num_tracklets = sum([len(sublist) for sublist in self.tracks_distributed_to_grids])

        if num_tracklets <= 0:
            return

        for list_agents_in_grid in self.tracks_distributed_to_grids:
            for current_tracklet in list_agents_in_grid:
                # Note: only rotate or correct for dynamic agents
                if not current_tracklet.is_dynamic:
                    continue
                w, h = current_tracklet.prev_center

                # Note: apply negative rotation angle, negative corrections
                w, h = stablization.apply_correction(w, h, center_point_w, center_point_h, -corr_rot, -corr_x, -corr_y)

                current_tracklet.prev_center = np.array([w, h])

    def initial_lane_polygon_adjustment(self, data):
        if STABLIZATION_SYSTEM_NAME not in data or data[STABLIZATION_SYSTEM_NAME] is None:
            return

        stablization = data[STABLIZATION_SYSTEM_NAME]
        stablization.initial_lane_polygon_adjustment(self.initial_rot_angle, self.initial_x, self.initial_y)

    def check_road_feature_static_tracklets(self, data):
 
        if STABLIZATION_SYSTEM_NAME not in data or data[STABLIZATION_SYSTEM_NAME] is None:
            return

        stablization = data[STABLIZATION_SYSTEM_NAME]
 
        num_tracklets = sum([len(sublist) for sublist in self.tracks_distributed_to_grids])
 
        if num_tracklets <= 0:
            return

        for list_agents_in_grid in self.tracks_distributed_to_grids:
            for current_tracklet in list_agents_in_grid:
                if current_tracklet.is_dynamic:
                    continue

                stablization.is_a_hit(current_tracklet.prev_center,
                                      current_tracklet.initial_aabb_width,
                                      current_tracklet.initial_aabb_height,
                                      current_tracklet.frame_counter)

    def compute_relative_lane_slot_measurements(self, data):

        if CONFIG_NAME not in data:
            return

        config = data[CONFIG_NAME]

        frame_number = data[FRAME_NUMBER_NAME]
        tracks_distributed_to_grids = data[TRACKED_AGENTS_NAME]
        video_name = data[VIDEO_NAME]

        width_authentic_resolution_roi, height_authentic_resolution_roi,\
            width_starting_point_authentic_resolution_roi, height_starting_point_authentic_resolution_roi,\
            width_tracking, height_tracking = data[DIMENSIONS_NAME]

        dict_lane_id = config.compute_lane_slot_id(video_name, frame_number,
                                                tracks_distributed_to_grids,
                                                width_starting_point_authentic_resolution_roi,
                                                height_starting_point_authentic_resolution_roi,
                                                width_authentic_resolution_roi /
                                                width_tracking,
                                                height_authentic_resolution_roi /
                                                height_tracking)

        data[RELATIVE_MEAS_NAME] = dict_lane_id

    def update_existing_and_create_new_tracklets_using_detection(self, data):

        if PROCESSED_PREDICTIONS_NAME not in data:
            return

        frame_number = data[FRAME_NUMBER_NAME]
        time_to_beginning = frame_number / self.fps
        frame_tracking = data[TRACKING_IMAGE_NAME]
        frame_color_tracking = data[TRACKING_IMAGE_RGB_NAME]
        list_polygons_registration_zone, list_polygons_unsubscription_zone, list_occ = data[CALIBRATION_ZONE_NAME]

        predictions = data[PROCESSED_PREDICTIONS_NAME]

        # Note: this is for plotting, each tracklet has its unique color code
        matched_index_of_detected_agents = {}

        for index_det, proc_pre_dict in predictions.items():
            index_grid_cell, index_col, index_row = proc_pre_dict[PROCESSED_PREDICTIONS_POS_INDEX_NAME]
            is_agent_dynamic = proc_pre_dict[PROCESSED_PREDICTIONS_DYNAMIC_NAME]
            agent_class = proc_pre_dict[PROCESSED_PREDICTIONS_CLASS_NAME]
            box_center = proc_pre_dict[PROCESSED_PREDICTIONS_CENTER_NAME]
            bbox_width = proc_pre_dict[PROCESSED_PREDICTIONS_BOX_WIDTH_NAME]
            bbox_height = proc_pre_dict[PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME]
            bounding_polygon = proc_pre_dict[PROCESSED_PREDICTIONS_POLYGON_NAME]
            polygon_length = proc_pre_dict[PROCESSED_PREDICTIONS_LENGTH_NAME]
            polygon_width = proc_pre_dict[PROCESSED_PREDICTIONS_WIDTH_NAME]
            color_mask_detection = proc_pre_dict[PROCESSED_PREDICTIONS_MASK_DET_NAME]
            color_mask_tracking = proc_pre_dict[PROCESSED_PREDICTIONS_MASK_TRACK_NAME]
            larger_principal_axis_point_1, larger_principal_axis_point_2, smaller_principal_axis_point_1,\
                smaller_principal_axis_point_2, larger_principal_axis_vector, smaller_principal_axis_vector,\
                offset_corner_1, offset_corner_2, offset_corner_3, offset_corner_4\
                = proc_pre_dict[PROCESSED_PREDICTIONS_EIGENVECTOR_NAME]
            feature_point = proc_pre_dict[PROCESSED_PREDICTIONS_FEATURE_POINTS_NAME]

            is_matched_existing_tracklet = False
            is_overlapped_existing_tracklet = False

            # Note: start to connect detection to existing tracklet, matching....

            # TODO remember, corner cases when tracklets are just at the boundary of the grids
            list_tracklets_to_check = self.tracks_distributed_to_grids[index_grid_cell]

            for index_obj in range(len(list_tracklets_to_check)):
                tracklet = list_tracklets_to_check[index_obj]

                # Note property of being dynamic or static has to be the same for checking matched tracklets
                if tracklet.is_dynamic != is_agent_dynamic:
                    continue

                is_to_update, is_overlap = tracklet.is_feature_update_qualified(box_center, bbox_width, bbox_height,
                                                                                bounding_polygon, polygon_length,
                                                                                polygon_width, frame_number, index_det)

                is_overlapped_existing_tracklet |= is_overlap

                if is_to_update:
                    tracklet.update_feature(
                        agent_class,
                        color_mask_detection, color_mask_tracking, box_center, polygon_length,
                        polygon_width,
                        bbox_width, bbox_height,
                        frame_number, larger_principal_axis_point_1,
                        larger_principal_axis_point_2, smaller_principal_axis_point_1,
                        smaller_principal_axis_point_2, larger_principal_axis_vector,
                        smaller_principal_axis_vector, offset_corner_1, offset_corner_2,
                        offset_corner_3, offset_corner_4, feature_point)

                    is_matched_existing_tracklet = True
                    matched_index_of_detected_agents[index_det] = tracklet
                    break

            is_new_obj = not (is_matched_existing_tracklet or is_overlapped_existing_tracklet)

            # Note: handle static object tracking differently
            if is_agent_dynamic is False:
                if is_new_obj and StaticObjectTracker.global_static_object_id < \
                        StaticObjectTracker.global_static_max_num_lane_marker:
                    static_tracklet = StaticObjectTracker(list_occ, frame_tracking, color_mask_detection, color_mask_tracking,
                                                          box_center, polygon_length, polygon_width,
                                                          bbox_width, bbox_height,
                                                          self.fps,
                                                          agent_class, self.num_row, self.num_col,
                                                          frame_number, self.feature_detector, frame_color_tracking)
                    if static_tracklet.is_initialized:
                        static_tracklet.register(index_det)
                        self.tracks_distributed_to_grids[static_tracklet.index_grid_cell].append(static_tracklet)
                continue

            # Note: from now on only handle dynamic agents, registration zone
            is_in_registration_zone = False
            is_in_unsubscription_zone = False

            if is_agent_dynamic and is_new_obj:
                point = geometry.Point(box_center[0], box_center[1])

                for poly in list_polygons_registration_zone:
                    if poly.contains(point):
                        is_in_registration_zone = True
                        break

                for poly in list_polygons_unsubscription_zone:
                    if poly.contains(point):
                        is_in_unsubscription_zone = True
                        break

            # Note: a little complicated logic to decide whether or not to register a new agent
            # (a) new agents means not matched to any existing agents in tracking system
            # (b) have to be in registration zone, or the first few frames of video where we register all agents
            # (c) notice, caveat: for the first few frames, agents could also be in unsubscription zone, we need to
            # ban those folks from registration....

            is_to_register = is_new_obj \
                and (is_in_registration_zone or
                     (time_to_beginning < self.initial_time_window_for_reg and not is_in_unsubscription_zone))

            if is_to_register:
                new_tracklet = ObjectTracker(self.cur_id, list_occ, frame_tracking, color_mask_detection, color_mask_tracking,
                                             box_center, polygon_length, polygon_width, bbox_width, bbox_height,
                                             self.fps, self.av_list,
                                             agent_class, self.num_row,
                                             self.num_col, larger_principal_axis_point_1, larger_principal_axis_point_2,
                                             smaller_principal_axis_point_1, smaller_principal_axis_point_2,
                                             larger_principal_axis_vector,
                                             smaller_principal_axis_vector, offset_corner_1, offset_corner_2,
                                             offset_corner_3, offset_corner_4,
                                             frame_number, feature_point, self.feature_detector, frame_color_tracking)
                if new_tracklet.is_initialized:
                    self.cur_id += 1
                    new_tracklet.register(index_det)
                    self.tracks_distributed_to_grids[new_tracklet.index_grid_cell].append(new_tracklet)

        data[MATCHED_AGENTS_NAME] = matched_index_of_detected_agents

    def update_existing_tracklets_using_image_tracking(self, data):
        num_tracklets = sum([len(sublist) for sublist in self.tracks_distributed_to_grids])
        if num_tracklets <= 0:
            data[TRACKED_AGENTS_NAME] = self.tracks_distributed_to_grids
            return

        frame_number = data[FRAME_NUMBER_NAME]
        frame_tracking = data[TRACKING_IMAGE_NAME]
        frame_color_tracking = data[TRACKING_IMAGE_RGB_NAME]
        list_polygons_registration_zone, list_polygons_unsubscription_zone, list_occ = data[CALIBRATION_ZONE_NAME]

        for list_agents_in_grid in self.tracks_distributed_to_grids:
            for current_tracklet in list_agents_in_grid:
                prev_index = current_tracklet.index_grid_cell
                if current_tracklet.is_dynamic and len(list_polygons_unsubscription_zone) > 0:
                    should_unsub = False
                    tracklet_latest_position = geometry.Point(current_tracklet.traj[-1])

                    for poly_unsub in list_polygons_unsubscription_zone:
                        if poly_unsub.contains(tracklet_latest_position):
                            current_tracklet.unregister(TRACKING_LOST_TYPE_UNSUB_ZONE)
                            list_agents_in_grid.remove(current_tracklet)
                            should_unsub = True
                            break
                    if should_unsub:
                        continue

                if current_tracklet.is_dynamic and current_tracklet.not_seen_in_detector(frame_number):
                    current_tracklet.unregister(TRACKING_LOST_TYPE_STALED_DET)
                    list_agents_in_grid.remove(current_tracklet)
                    continue

                status = current_tracklet.track(frame_tracking, self.use_homography, self.use_ransac,
                                                frame_number, frame_color_tracking)

                if status is False:
                    list_agents_in_grid.remove(current_tracklet)
                    continue

                cur_index = current_tracklet.index_grid_cell
                if cur_index != prev_index:
                    self.tracks_distributed_to_grids[prev_index].remove(current_tracklet)
                    self.tracks_distributed_to_grids[cur_index].append(current_tracklet)

        data[TRACKED_AGENTS_NAME] = self.tracks_distributed_to_grids

    def report_statistics(self):
        num_total, num_miss = ObjectTracker.global_total_registered_agents,\
                              ObjectTracker.global_total_agents_not_reach_unsub_zone
        num_tracked = num_total - num_miss
        succ_rate = num_tracked * 1. / num_total * 100.

        logging.info("track_agents_individualized: System Success Rate: %.1f [%%] %d/%d" %
                     (succ_rate, num_tracked, num_total))

    def cleanup(self):
        super().cleanup()

        if ObjectTracker.global_num_good_feature_to_track > 0:
            logging.debug("time spent on GFTT feature detection %.2f [s] for %s times, average is %.4f [s]" %
                  (ObjectTracker.global_time_good_feature_to_track, ObjectTracker.global_num_good_feature_to_track,
                   ObjectTracker.global_time_good_feature_to_track / ObjectTracker.global_num_good_feature_to_track))

        if ObjectTracker.global_num_orb > 0:
            logging.debug("time spent on ORB feature detection %.2f [s] for %s times, average is %.4f [s]" %
                  (ObjectTracker.global_time_orb, ObjectTracker.global_num_orb,
                   ObjectTracker.global_time_orb / ObjectTracker.global_num_orb))

        if ObjectTracker.global_num_dnn_feature_points > 0:
            logging.debug("time spent on DNN feature detection %.2f [s] for %s times, average is %.4f [s]" %
                  (ObjectTracker.global_time_dnn_feature_points, ObjectTracker.global_num_dnn_feature_points,
                   ObjectTracker.global_time_dnn_feature_points / ObjectTracker.global_num_dnn_feature_points))

        if ObjectTracker.global_num_calculate_optical_flow > 0:
            logging.debug("time spent on Optical FLow computing %.2f [s] for %s times, average is %.4f [s]" %
                  (ObjectTracker.global_time_calculate_optical_flow, ObjectTracker.global_num_calculate_optical_flow,
                   ObjectTracker.global_time_calculate_optical_flow / ObjectTracker.global_num_calculate_optical_flow))

        if ObjectTracker.global_num_estimate_affine > 0:
            logging.debug("time spent on estimating affine %.2f [s] for %s times, average is %.4f [s]" %
                  (ObjectTracker.global_time_estimate_affine, ObjectTracker.global_num_estimate_affine,
                   ObjectTracker.global_time_estimate_affine / ObjectTracker.global_num_estimate_affine))

        self.report_statistics()


