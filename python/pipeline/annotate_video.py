import timeit
import cv2
import numpy as np

try:
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
except:
    pass

from utility_for_plotting import plot_lane_slot_polygons, plot_lane_id, plot_frame_id
from add_rr_logo import add_rr_logo
from user_selected_roi import plot_roi_zone_polygon, plot_roi_zone, plot_center
from pipeline.pipeline import Pipeline
from constant_values import *


class AnnotateVideo(Pipeline):
    def __init__(self, scale_width_from_tracking_to_output, scale_height_from_tracking_to_output,
                 is_plot_polygon_requested, is_plot_box_req, is_plot_mas_requested, road_markers,
                 lane_id_slot_id_measurements, reg_unsub_zone, is_tracklet_traj_plot_req,
                 is_tracking_optical_flow_feature_point_plot_req,
                 is_plot_stablization_zone_req,
                 is_plot_stablization_corrections_req, output_video_width, output_video_height, top_right, logo_pos,
                 plot_warped_image, plot_car_heading, plot_feature_points, is_plot_resolution, center_dot_size,
                 line_width, agent_id_text_size, agent_id_font_size, agent_id_offset_w, agent_id_offset_h):
        self.scale_width_from_tracking_to_output = scale_width_from_tracking_to_output
        self.scale_height_from_tracking_to_output = scale_height_from_tracking_to_output

        self.default_color_code_full_scale = [int(v * 255) for v in (1, 1, 1)]

        # Note: what is this auto_config switch doing here, we set it to true so that static objects can be plotted
        # i change my mind... road markers are static obojects and they are mostly used for stablization corrections
        # if i switch off plot stablization zone, then do not plot statc objects
        # self.auto_config = True

        self.is_plot_polygon_requested = is_plot_polygon_requested
        self.is_plot_box_req = is_plot_box_req
        self.is_plot_mask_requested = is_plot_mas_requested
        self.is_plot_resolution = is_plot_resolution
        self.road_markers = road_markers
        self.lane_id_slot_id_measurements = lane_id_slot_id_measurements
        self.reg_unsub_zone = reg_unsub_zone

        self.output_video_width = output_video_width
        self.output_video_height = output_video_height

        self.top_right = top_right
        self.logo_pos = logo_pos

        self.cached_processed_predictions = None
        self.cached_matched_agents_in_tracking_system = None

        self.plot_warped_image = plot_warped_image
        self.plot_car_heading = plot_car_heading
        self.plot_feature_points = plot_feature_points

        self.is_plot_stablization_zone_req = is_plot_stablization_zone_req
        self.is_plot_stablization_corrections_req = is_plot_stablization_corrections_req

        self.is_tracklet_traj_plot_req = is_tracklet_traj_plot_req
        self.is_tracking_optical_flow_feature_point_plot_req = is_tracking_optical_flow_feature_point_plot_req

        self.center_dot_size = int(center_dot_size)
        self.line_width = int(line_width)
        self.agent_id_text_size = int(agent_id_text_size)
        self.agent_id_font_size = float(agent_id_font_size)
        self.agent_id_offset_w = int(agent_id_offset_w)
        self.agent_id_offset_h = int(agent_id_offset_h)

        super().__init__("AnnotateVideo")

    def map(self, data):
        start_time = timeit.default_timer()

        if PROCESSED_PREDICTIONS_NAME in data:
            processed_predictions = data[PROCESSED_PREDICTIONS_NAME]

            if self.is_plot_polygon_requested:
                self.annotate_polygon_boundary(processed_predictions, data)
                self.cached_processed_predictions = processed_predictions

            if self.is_plot_box_req:
                self.annotate_aabb(processed_predictions, data)
                self.cached_processed_predictions = processed_predictions

            if self.is_plot_mask_requested:
                self.annotate_semantic_map(processed_predictions, data)
                self.cached_processed_predictions = processed_predictions

        # Note when this frame has no detections, we plot the previously cached results
        else:
            if self.is_plot_polygon_requested:
                self.annotate_polygon_boundary(self.cached_processed_predictions, data, using_cache=True)

            if self.is_plot_box_req:
                self.annotate_aabb(self.cached_processed_predictions, data, using_cache=True)

            if self.is_plot_mask_requested:
                self.annotate_semantic_map(self.cached_processed_predictions, data, using_cache=True)

        if self.is_tracklet_traj_plot_req:
            self.annotate_tracking_trajectory(data)

        if self.is_tracking_optical_flow_feature_point_plot_req:
            self.annotate_tracking_optical_flow_feature_points(data)

        if self.road_markers:
            self.annotate_calibrations(data)

        if self.reg_unsub_zone:
            self.annotate_calibration_zones(data)

        if self.lane_id_slot_id_measurements:
            self.annotate_lane_id_slot_id(data)

        if self.plot_warped_image:
            self.warp_image(data)

        if self.is_plot_stablization_zone_req:
            self.annotate_stablization_zone(data)

        if self.is_plot_stablization_corrections_req:
            self.annotate_stablization_correction(data)

        if self.is_plot_resolution:
            self.annotate_calib_points(data)

        # TODO remove logo annotation in the future
        # self.annotate_logo(data)

        self.annotate_frame_id(data)

        self.annotate_tracking_statistics(data)

        self.timer += timeit.default_timer() - start_time

        return data

    def warp_image(self, data):
        output_image = data[OUTPUT_IMAGE_NAME]
        image_mapping = data[CALIBRATION_IMAGE_MAPPING_NAME]
        output_image = image_mapping.transform(output_image)
        output_image = cv2.resize(output_image, (self.output_video_width, self.output_video_height))
        data[OUTPUT_IMAGE_NAME] = output_image

    def annotate_tracking_statistics(self, data):
        if TRACKING_STATISTICS not in data:
            return

        num_total, num_miss = data[TRACKING_STATISTICS]
        num_tracked = num_total - num_miss
        succ_rate = num_tracked * 1. / num_total * 100.

        frame_output = data[OUTPUT_IMAGE_NAME]

        height, width = frame_output.shape[0], frame_output.shape[1]
        scale_h = 0.15
        scale_w = 0.15
        plot_height = int(height * scale_h)
        plot_width = int(width * scale_w)

        cv2.putText(frame_output, "System Success Rate: %.1f [%%] %d/%d" % (succ_rate, num_tracked, num_total),
                    (plot_width, plot_height), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)

    def annotate_frame_id(self, data):
        output_image = data[OUTPUT_IMAGE_NAME]
        frame_counter = data[FRAME_NUMBER_NAME]
        plot_frame_id(output_image, frame_counter)

    def annotate_logo(self, data):
        output_image = data[OUTPUT_IMAGE_NAME]
        add_rr_logo(output_image, self.top_right, self.logo_pos)

    def annotate_tracking_optical_flow_feature_points(self, data):
        if TRACKED_AGENTS_NAME not in data:
            return

        tracks_distributed_to_grids = data[TRACKED_AGENTS_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]

        for list_agents_in_grid in tracks_distributed_to_grids:
            for current_tracklet in list_agents_in_grid:
                if current_tracklet.is_dynamic:
                    current_tracklet.plot_tracking_feature_poins(
                        frame_output,
                        self.scale_width_from_tracking_to_output,
                        self.scale_height_from_tracking_to_output,
                        self.center_dot_size)

    def annotate_semantic_map(self, processed_predictions, data, using_cache=False):

        if processed_predictions is None:
            return

        output_image = data[OUTPUT_IMAGE_NAME]
        height, width = output_image.shape[0], output_image.shape[1]

        empty_image = np.zeros((height, width, 3), dtype=np.uint8)
        # output_image = cv2.bitwise_and(output_image, output_image, mask=global_mask)

        list_of_mask_points = []
        list_of_mask_points_color_code = []

        color_code = self.default_color_code_full_scale

        for index, dict_processed in processed_predictions.items():

            if using_cache is True:
                matched_index_of_detected_agents = self.cached_matched_agents_in_tracking_system
                if matched_index_of_detected_agents is not None and index in matched_index_of_detected_agents:
                    tracklet = matched_index_of_detected_agents[index]
                    color_code = tracklet.color_code
            else:
                if MATCHED_AGENTS_NAME in data:
                    matched_index_of_detected_agents = data[MATCHED_AGENTS_NAME]

                    self.cached_matched_agents_in_tracking_system = matched_index_of_detected_agents

                    if index in matched_index_of_detected_agents:
                        tracklet = matched_index_of_detected_agents[index]
                        color_code = tracklet.color_code

            mask_in_tracking_space = dict_processed[PROCESSED_PREDICTIONS_MASK_TRACK_NAME]

            mask_in_tracking_space_boolean = mask_in_tracking_space > 250

            seg_map = SegmentationMapsOnImage(mask_in_tracking_space_boolean,
                                              shape=mask_in_tracking_space_boolean.shape)
            seg_map = seg_map.resize((height, width))
            scaled_mask = seg_map.get_arr()

            mask_indices = np.argwhere(scaled_mask == True).astype(np.float)
            mask_indices = mask_indices.astype(np.int)

            for point in mask_indices:
                list_of_mask_points.append((point[1], point[0]))
                list_of_mask_points_color_code.append(color_code)

            # global_mask = cv2.bitwise_or(global_mask, global_mask, mask=scaled_mask)

        for i_point in range(len(list_of_mask_points)):
            _point = list_of_mask_points[i_point]
            cv2.circle(empty_image, tuple(_point), 1, list_of_mask_points_color_code[i_point], -1)

        output_image = cv2.addWeighted(output_image, 0.5, empty_image, 0.5, 0)
        data[OUTPUT_IMAGE_NAME] = output_image

    def annotate_aabb(self, processed_predictions, data, using_cache=False):
        output_image = data[OUTPUT_IMAGE_NAME]
        for index, dict_processed in processed_predictions.items():
            center_x, center_y = dict_processed[PROCESSED_PREDICTIONS_CENTER_NAME]
            width = dict_processed[PROCESSED_PREDICTIONS_BOX_WIDTH_NAME]
            height = dict_processed[PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME]
            agent_class = dict_processed[PROCESSED_PREDICTIONS_CLASS_NAME]

            # Note: 1 -> 2 -> 3 -> 4 tl -> tr -> br -> bl
            x_corner_1, y_corner_1 = center_x - 0.5 * width, center_y - 0.5 * height
            x_corner_2, y_corner_2 = center_x + 0.5 * width, center_y - 0.5 * height
            x_corner_3, y_corner_3 = center_x + 0.5 * width, center_y + 0.5 * height
            x_corner_4, y_corner_4 = center_x - 0.5 * width, center_y + 0.5 * height
            x_corner_1 = int(x_corner_1 * self.scale_width_from_tracking_to_output)
            x_corner_2 = int(x_corner_2 * self.scale_width_from_tracking_to_output)
            x_corner_3 = int(x_corner_3 * self.scale_width_from_tracking_to_output)
            x_corner_4 = int(x_corner_4 * self.scale_width_from_tracking_to_output)
            y_corner_1 = int(y_corner_1 * self.scale_height_from_tracking_to_output)
            y_corner_2 = int(y_corner_2 * self.scale_height_from_tracking_to_output)
            y_corner_3 = int(y_corner_3 * self.scale_height_from_tracking_to_output)
            y_corner_4 = int(y_corner_4 * self.scale_height_from_tracking_to_output)

            color_code = self.default_color_code_full_scale

            if using_cache is True:
                matched_index_of_detected_agents = self.cached_matched_agents_in_tracking_system
                if matched_index_of_detected_agents is not None and index in matched_index_of_detected_agents:
                    tracklet = matched_index_of_detected_agents[index]
                    color_code = tracklet.color_code
            else:
                if MATCHED_AGENTS_NAME in data:
                    # Note: matched_index_of_detected_agents is a dict that matches det index to tracklet
                    matched_index_of_detected_agents = data[MATCHED_AGENTS_NAME]

                    self.cached_matched_agents_in_tracking_system = matched_index_of_detected_agents

                    if index in matched_index_of_detected_agents:
                        tracklet = matched_index_of_detected_agents[index]
                        color_code = tracklet.color_code

            cv2.line(output_image, (x_corner_1, y_corner_1), (x_corner_2, y_corner_2), color_code, 4)
            cv2.line(output_image, (x_corner_2, y_corner_2), (x_corner_3, y_corner_3), color_code, 4)
            cv2.line(output_image, (x_corner_3, y_corner_3), (x_corner_4, y_corner_4), color_code, 4)
            cv2.line(output_image, (x_corner_4, y_corner_4), (x_corner_1, y_corner_1), color_code, 4)

            cv2.putText(output_image, "det_%s %s" % (index, agent_class),
                        (x_corner_1, y_corner_1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        BLUE,
                        2)

    def annotate_polygon_boundary(self, processed_predictions, data, using_cache=False):
        return
        output_image = data[OUTPUT_IMAGE_NAME]

        for index, dict_processed in processed_predictions.items():
            raw_poly = dict_processed[PROCESSED_PREDICTIONS_POLY_POINTS_NAME]
            corner_1, corner_2, corner_3, corner_4 = raw_poly
            x_corner_1, y_corner_1 = corner_1
            x_corner_2, y_corner_2 = corner_2
            x_corner_3, y_corner_3 = corner_3
            x_corner_4, y_corner_4 = corner_4
            x_corner_1 = int(x_corner_1 * self.scale_width_from_tracking_to_output)
            x_corner_2 = int(x_corner_2 * self.scale_width_from_tracking_to_output)
            x_corner_3 = int(x_corner_3 * self.scale_width_from_tracking_to_output)
            x_corner_4 = int(x_corner_4 * self.scale_width_from_tracking_to_output)
            y_corner_1 = int(y_corner_1 * self.scale_height_from_tracking_to_output)
            y_corner_2 = int(y_corner_2 * self.scale_height_from_tracking_to_output)
            y_corner_3 = int(y_corner_3 * self.scale_height_from_tracking_to_output)
            y_corner_4 = int(y_corner_4 * self.scale_height_from_tracking_to_output)

            color_code = self.default_color_code_full_scale

            if using_cache is True:
                matched_index_of_detected_agents = self.cached_matched_agents_in_tracking_system
                if matched_index_of_detected_agents is not None and index in matched_index_of_detected_agents:
                    tracklet = matched_index_of_detected_agents[index]
                    color_code = tracklet.color_code
            else:
                if MATCHED_AGENTS_NAME in data:
                    matched_index_of_detected_agents = data[MATCHED_AGENTS_NAME]

                    self.cached_matched_agents_in_tracking_system = matched_index_of_detected_agents

                    if index in matched_index_of_detected_agents:
                        tracklet = matched_index_of_detected_agents[index]
                        color_code = tracklet.color_code

            cv2.line(output_image, (x_corner_1, y_corner_1), (x_corner_2, y_corner_2), color_code, 4)
            cv2.line(output_image, (x_corner_2, y_corner_2), (x_corner_3, y_corner_3), color_code, 4)
            cv2.line(output_image, (x_corner_3, y_corner_3), (x_corner_4, y_corner_4), color_code, 4)
            cv2.line(output_image, (x_corner_4, y_corner_4), (x_corner_1, y_corner_1), color_code, 4)

            cv2.putText(output_image, "det_%s" % index,
                        (x_corner_1, y_corner_1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        RED,
                        2)

    def annotate_tracking_trajectory(self, data):
        if TRACKED_AGENTS_NAME not in data:
            return

        tracks_distributed_to_grids = data[TRACKED_AGENTS_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]

        for list_agents_in_grid in tracks_distributed_to_grids:
            for current_tracklet in list_agents_in_grid:
                if (self.is_plot_stablization_zone_req and not current_tracklet.is_dynamic) or current_tracklet.is_dynamic:
                    current_tracklet.plot_traj(frame_output,
                        self.scale_width_from_tracking_to_output,
                        self.scale_height_from_tracking_to_output,
                        self.is_plot_polygon_requested,
                        self.plot_car_heading,
                        self.plot_feature_points,
                        self.center_dot_size, self.line_width, self.agent_id_text_size, self.agent_id_font_size,
                        self.agent_id_offset_w, self.agent_id_offset_h)

    def annotate_calibrations(self, data):
        if CONFIG_NAME not in data:
            return

        config = data[CONFIG_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]
        frame_counter = data[FRAME_NUMBER_NAME]
        width_authentic_resolution_roi,\
            height_authentic_resolution_roi,\
            width_starting_point_authentic_resolution_roi,\
            height_starting_point_authentic_resolution_roi, width_tracking, height_tracking =\
            data[DIMENSIONS_NAME]
        video_name = data[VIDEO_NAME]

        plot_lane_slot_polygons(frame_output, config, video_name,
                                frame_counter,
                                width_starting_point_authentic_resolution_roi,
                                height_starting_point_authentic_resolution_roi,
                                self.output_video_width / width_authentic_resolution_roi,
                                self.output_video_height / height_authentic_resolution_roi)

    def annotate_lane_id_slot_id(self, data):
        if CONFIG_NAME not in data:
            return

        frame_output = data[OUTPUT_IMAGE_NAME]
        dict_lane_id = data[RELATIVE_MEAS_NAME]
        width_authentic_resolution_roi, height_authentic_resolution_roi, width_starting_point_authentic_resolution_roi,\
            height_starting_point_authentic_resolution_roi, width_tracking, height_tracking = data[DIMENSIONS_NAME]
        tracks_distributed_to_grids = data[TRACKED_AGENTS_NAME]

        plot_lane_id(frame_output, dict_lane_id, tracks_distributed_to_grids,
                     self.output_video_width / width_tracking,
                     self.output_video_height / height_tracking)

    def annotate_calibration_zones(self, data):
        list_polygons_registration_zone, list_polygons_unsubscription_zone, list_occ = data[CALIBRATION_ZONE_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]
        _, _, _, _, width_tracking, height_tracking = data[DIMENSIONS_NAME]
        if len(list_polygons_registration_zone) > 0:
            plot_roi_zone_polygon(list_polygons_registration_zone,
                                  self.output_video_width,
                                  self.output_video_height,
                                  frame_output,
                                  width_tracking,
                                  height_tracking,
                                  BLUE)

        if len(list_polygons_unsubscription_zone) > 0:
            plot_roi_zone_polygon(list_polygons_unsubscription_zone,
                                  self.output_video_width,
                                  self.output_video_height,
                                  frame_output,
                                  width_tracking,
                                  height_tracking,
                                  BLUE)

        if len(list_occ) > 0:
            plot_roi_zone_polygon(list_occ,
                                  self.output_video_width,
                                  self.output_video_height,
                                  frame_output,
                                  width_tracking,
                                  height_tracking,
                                  YELLOW_RGB)

    def annotate_stablization_zone(self, data):
        if STABLIZATION_SYSTEM_NAME not in data:
            return

        stablization = data[STABLIZATION_SYSTEM_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]

        width_authentic_resolution_roi, \
        height_authentic_resolution_roi, \
        width_starting_point_authentic_resolution_roi, \
        height_starting_point_authentic_resolution_roi, width_tracking, height_tracking = \
            data[DIMENSIONS_NAME]

        plot_roi_zone_polygon(stablization.list_polygons_original,
                              self.output_video_width,
                              self.output_video_height,
                              frame_output,
                              width_tracking,
                              height_tracking,
                              YELLOW_RGB)

        plot_roi_zone_polygon(stablization.list_polygons,
                              self.output_video_width,
                              self.output_video_height,
                              frame_output,
                              width_tracking,
                              height_tracking,
                              BLUE)

        plot_center(stablization.center,
                    self.output_video_width,
                    self.output_video_height,
                    frame_output,
                    width_tracking,
                    height_tracking,
                    RED)

    def annotate_stablization_correction(self, data):
        if STABLIZATION_SYSTEM_NAME not in data:
            return

        stablization = data[STABLIZATION_SYSTEM_NAME]
        frame_output = data[OUTPUT_IMAGE_NAME]

        height, width = frame_output.shape[0], frame_output.shape[1]
        scale = 0.1
        scale_w = 0.85
        plot_height = int(height * scale)
        plot_width = int(width * scale_w)
        plot_h_interval = int(height * 0.15)

        cv2.putText(frame_output, "Drift parameters:",
                    (plot_width, plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    1)

        plot_height += plot_h_interval

        cv2.putText(frame_output, "Rot. angle %.1f [deg]" % stablization.correction_rot_angle,
                    (plot_width, plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    1)

        plot_height += plot_h_interval

        cv2.putText(frame_output, "X shift %.1f [pix]" % stablization.correction_shift_width,
                    (plot_width, plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    1)

        plot_height += plot_h_interval

        cv2.putText(frame_output, "Y shift %.1f [pix]" % stablization.correction_shift_height,
                    (plot_width, plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    1)

    def annotate_calib_points(self, data):
        if CALIBRATION_RESOLUTION not in data:
            return

        frame_output = data[OUTPUT_IMAGE_NAME]
        height, width = frame_output.shape[0], frame_output.shape[1]
        scale = 0.1
        scale_w = 0.05
        plot_height = int(height * scale)
        plot_width = int(width * scale_w)

        resolution = data[CALIBRATION_RESOLUTION]
        cv2.putText(frame_output,
                    "Resolution %.3f [m/pixel]" %
                    (
                        resolution),
                    (plot_width, plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    2)
