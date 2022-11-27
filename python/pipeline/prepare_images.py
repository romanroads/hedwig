import os
import timeit
import cv2 as cv
import logging
import numpy as np
from pathlib import Path

from pipeline.pipeline import Pipeline

from utils import get_video_file_artifact_file_name
from user_selected_mask import get_masked_image_with_roi, get_rect_roi
from user_selected_roi import get_roi, get_roi_using_json, get_road_marks, save_user_selected_roi
from interactive_calibration import interactive_calibration
from constant_values import *


class PrepareImages(Pipeline):
    def __init__(self, video_src, input_video_width, input_video_height, width_output, height_output, width_detection,
                 height_detection, width_tracking, height_tracking, auto_processing, calibrate, config_json,
                 stablization_zone_mannual_input, plot_vector_image, roi_manual_input, registration_zone_mannual_input,
                 unsubscription_zone_mannual_input):

        self.auto_processing = auto_processing
        width_output, height_output, width_detection, height_detection, width_tracking, height_tracking = \
            int(width_output), int(height_output), int(width_detection), int(height_detection), int(width_tracking),\
            int(height_tracking)

        self.config_json = config_json

        self.is_output_resize_requested = width_output != -1 and height_output != -1
        self.is_detection_resize_requested = width_detection != -1 and height_detection != -1
        self.is_tracking_resize_requested = width_tracking != -1 and height_tracking != -1

        self.input_video_width = input_video_width
        self.input_video_height = input_video_height

        self.width_output = input_video_width
        self.height_output = input_video_height

        self.width_detection = input_video_width
        self.height_detection = input_video_height

        self.width_tracking = input_video_width
        self.height_tracking = input_video_height

        if self.is_detection_resize_requested:
            self.width_detection = width_detection
            self.height_detection = height_detection

        if self.is_tracking_resize_requested:
            self.width_tracking = width_tracking
            self.height_tracking = height_tracking

        if self.is_output_resize_requested:
            self.width_output = width_output
            self.height_output = height_output

        self.scale_width_from_det_to_track = self.width_tracking / self.width_detection
        self.scale_height_from_det_to_track = self.height_tracking / self.height_detection
        self.scale_width_from_det_to_output = self.width_output / self.width_detection
        self.scale_height_from_det_to_output = self.height_output / self.height_detection
        self.scale_width_from_tracking_to_output = self.width_output / self.width_tracking
        self.scale_height_from_tracking_to_output = self.height_output / self.height_tracking

        dirname, _ = os.path.split(os.path.abspath(__file__))

        path_to_output = os.path.join(Path(dirname).parent.parent, PROCESSED_VIDEO_FOLDER_NAME)

        if os.path.exists(path_to_output) is False:
            os.mkdir(path_to_output)

        self.roi_file_path = path_to_output

        _, self.roi_file_name = get_video_file_artifact_file_name(video_src, "_ROI.txt")

        self.list_frac_pixel_cords_roi = []
        self.list_frac_pixel_cords_roi_registration_zone = []
        self.list_frac_pixel_cords_roi_unsubscription_zone = []
        self.list_frac_pixel_cords_roi_occlusion_zone = []
        self.list_frac_pixel_cords_roi_stablization_zone = []
        self.list_polygons_registration_zone = []
        self.list_polygons_unsubscription_zone = []
        self.list_polygons_occlusion_zone = []
        self.list_polygons_stalization_zone = []

        self.width_authentic_resolution_roi = input_video_width
        self.height_authentic_resolution_roi = input_video_height
        self.width_starting_point_authentic_resolution_roi = 0
        self.height_starting_point_authentic_resolution_roi = 0

        self.timer_mask_image_on_roi = 0.
        self.num_mask_image_on_roi = 0

        self.rect_roi = None
        self.calibrate = calibrate
        self.last_frame_skipped_from_calibration = True

        self.user_selected_road_marks = None

        self.stablization_zone_mannual_input = stablization_zone_mannual_input
        self.stablization_zone_mannual_input_list = None
        self.plot_vector_image = plot_vector_image
        
        if len(self.stablization_zone_mannual_input) > 0:
            try:
                list_zones = self.stablization_zone_mannual_input.split(";")
                self.stablization_zone_mannual_input_list = []
                for zone in list_zones:
                    if len(zone) <= 0:
                        continue
                    x_y_coords = zone.split(",")
                    x_y_coords_list = list(np.array(x_y_coords, dtype=float))
                    x_y_coords_pair_list = []
                    for i in range(int(len(x_y_coords_list) / 2)):
                        x_y_coords_pair_list.append((x_y_coords_list[2 * i], x_y_coords_list[2 * i + 1]))
                    self.stablization_zone_mannual_input_list.append(x_y_coords_pair_list)
            except:
                self.stablization_zone_mannual_input_list = None

        self.roi_mannual_input = roi_manual_input
        self.roi_mannual_input_list = None
        if len(self.roi_mannual_input) > 0:
            try:
                self.roi_mannual_input_list = []
                x_y_coords = self.roi_mannual_input.split(",")
                x_y_coords_list = list(np.array(x_y_coords, dtype=float))
                for i in range(int(len(x_y_coords_list) / 2)):
                    self.roi_mannual_input_list.append((x_y_coords_list[2 * i], x_y_coords_list[2 * i + 1]))
            except:
                self.roi_mannual_input_list = None

        self.registration_zone_mannual_input = registration_zone_mannual_input
        self.registration_zone_mannual_input_list = None
        if len(self.registration_zone_mannual_input) > 0:
            try:
                list_zones = self.registration_zone_mannual_input.split(";")
                self.registration_zone_mannual_input_list = []
                for zone in list_zones:
                    if len(zone) <= 0:
                        continue
                    x_y_coords = zone.split(",")
                    x_y_coords_list = list(np.array(x_y_coords, dtype=float))
                    x_y_coords_pair_list = []
                    for i in range(int(len(x_y_coords_list) / 2)):
                        x_y_coords_pair_list.append((x_y_coords_list[2 * i], x_y_coords_list[2 * i + 1]))
                    self.registration_zone_mannual_input_list.append(x_y_coords_pair_list)
            except:
                self.registration_zone_mannual_input_list = None

        self.unsubscription_zone_mannual_input = unsubscription_zone_mannual_input
        self.unsubscription_zone_mannual_input_list = None
        if len(self.unsubscription_zone_mannual_input) > 0:
            try:
                list_zones = self.unsubscription_zone_mannual_input.split(";")
                self.unsubscription_zone_mannual_input_list = []
                for zone in list_zones:
                    if len(zone) <= 0:
                        continue
                    x_y_coords = zone.split(",")
                    x_y_coords_list = list(np.array(x_y_coords, dtype=float))
                    x_y_coords_pair_list = []
                    for i in range(int(len(x_y_coords_list) / 2)):
                        x_y_coords_pair_list.append((x_y_coords_list[2 * i], x_y_coords_list[2 * i + 1]))
                    self.unsubscription_zone_mannual_input_list.append(x_y_coords_pair_list)
            except:
                self.unsubscription_zone_mannual_input_list = None

        super().__init__("PrepareImages")

    def map(self, data):
        start_time = timeit.default_timer()

        image = data[RAW_IMAGE_NAME]
        frame_counter = data[FRAME_NUMBER_NAME]

        if frame_counter == 0:
            self.list_frac_pixel_cords_roi, self.list_frac_pixel_cords_roi_registration_zone, \
                self.list_frac_pixel_cords_roi_unsubscription_zone, \
                self.list_frac_pixel_cords_roi_occlusion_zone, \
                self.list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected = \
                get_roi(image, self.roi_file_path, self.roi_file_name, self.auto_processing)

            # Note: this line below will pop out UI for users to point & click the stablization zones
            # if auto_processing is True, then this line below will return empty list of stablization zones
            # self.user_selected_road_marks = get_road_marks(image, self.auto_processing)
            self.user_selected_road_marks = self.list_frac_pixel_cords_roi_stablization_zone

            if self.stablization_zone_mannual_input_list is not None and\
                len(self.stablization_zone_mannual_input_list) > 0:
                logging.info("prepare_images: using manually input stablization zone: %s" %\
                             self.stablization_zone_mannual_input_list)
                self.user_selected_road_marks = self.stablization_zone_mannual_input_list

            # Note: if user mannually input a ROI, we give this higher priority than config json downloaded from
            # calib object from cloud
            if self.roi_mannual_input_list is not None:
                self.list_frac_pixel_cords_roi = self.roi_mannual_input_list

            if self.registration_zone_mannual_input_list is not None:
                self.list_frac_pixel_cords_roi_registration_zone = self.registration_zone_mannual_input_list

            if self.unsubscription_zone_mannual_input_list is not None:
                self.list_frac_pixel_cords_roi_unsubscription_zone = self.unsubscription_zone_mannual_input_list

            self.rect_roi = get_rect_roi(image, self.list_frac_pixel_cords_roi)

            r_x, r_y, r_w, r_h = self.rect_roi

            logging.debug("prepare_images: user selected ROI using: x_start [horizontal] %s, "
                  "y_start [vertical] %s with resolution W x H (%s x %s)" % (r_x, r_y, r_w, r_h))

            self.width_authentic_resolution_roi = r_w
            self.height_authentic_resolution_roi = r_h
            self.width_starting_point_authentic_resolution_roi = r_x
            self.height_starting_point_authentic_resolution_roi = r_y

            logging.debug("prepare_images: even though ROI w %s x h %s is selected, but the actual input layer to "
                  "DNN is still w %s x h %s" % (r_w, r_h, self.width_detection, self.height_detection))

            if not is_everything_not_selected:
                save_user_selected_roi(self.roi_file_path, self.roi_file_name, self.list_frac_pixel_cords_roi,
                                       self.list_frac_pixel_cords_roi_registration_zone,
                                       self.list_frac_pixel_cords_roi_occlusion_zone,
                                       self.list_frac_pixel_cords_roi_unsubscription_zone,
                                       self.list_frac_pixel_cords_roi_stablization_zone)

        start_time_mask_image = timeit.default_timer()
        image = get_masked_image_with_roi(image, self.list_frac_pixel_cords_roi, self.rect_roi)
        self.timer_mask_image_on_roi += timeit.default_timer() - start_time_mask_image
        self.num_mask_image_on_roi += 1

        if self.calibrate and self.last_frame_skipped_from_calibration:
            self.last_frame_skipped_from_calibration = interactive_calibration(image,
                self.width_starting_point_authentic_resolution_roi,
                 self.height_starting_point_authentic_resolution_roi,
                 self.width_tracking, self.height_tracking,
                 self.input_video_width, self.input_video_height)

        # Note this resizing is for DNN input feed
        image_detection = image
        if self.is_detection_resize_requested:
            image_detection = cv.resize(image, (self.width_detection, self.height_detection))
        data[DETECTION_IMAGE_NAME] = image_detection

        image_output = image
        if self.is_output_resize_requested:
            image_output = cv.resize(image, (self.width_output, self.height_output))

        data[OUTPUT_IMAGE_NAME] = image_output

        if self.plot_vector_image:
            data[OUTPUT_IMAGE_NAME] = np.zeros_like(image_output)

        if self.is_tracking_resize_requested:
            image_tracking = cv.cvtColor(cv.resize(image, (self.width_tracking,
                                                           self.height_tracking)), cv.COLOR_BGR2GRAY)
            image_tracking_rgb = cv.resize(image, (self.width_tracking, self.height_tracking))
        else:
            image_tracking = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_tracking_rgb = image

        data[TRACKING_IMAGE_NAME] = image_tracking

        data[TRACKING_IMAGE_RGB_NAME] = image_tracking_rgb

        data[DIMENSIONS_NAME] = [
            self.width_authentic_resolution_roi,
            self.height_authentic_resolution_roi,
            self.width_starting_point_authentic_resolution_roi,
            self.height_starting_point_authentic_resolution_roi,
            self.width_tracking,
            self.height_tracking
        ]
        data[CALIBRATION_ZONE_POINTS_NAME] = [self.list_frac_pixel_cords_roi_registration_zone,
                                              self.list_frac_pixel_cords_roi_unsubscription_zone,
                                              self.list_frac_pixel_cords_roi_occlusion_zone]

        data[STABLIZATION_ZONE_NAME] = self.user_selected_road_marks

        self.timer += timeit.default_timer() - start_time
        return data

    def cleanup(self):
        super(PrepareImages, self).cleanup()
        if self.num_mask_image_on_roi > 0:
            logging.debug("prepare_images: time spent on masking image from ROI %.2f [s] for %s times, "
                          "average is %.4f [s]" %
                  (self.timer_mask_image_on_roi, self.num_mask_image_on_roi,
                   self.timer_mask_image_on_roi / self.num_mask_image_on_roi))
