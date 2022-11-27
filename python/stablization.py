import numpy as np
from shapely import geometry
from utils import is_overlap
import logging


class Stablization(object):
    def __init__(self, config, video_file):
        self.list_frac_pixel_cords = []
        self.list_polygons = []
        self.list_polygons_original = []
        self.list_polygons_latest = []

        # TODO here list of widths and height is for AABB box, width is horizontal axis and height is veticle axis on image
        self.list_widths = []
        self.list_heights = []

        self.list_box_short_radius = []
        self.list_box_long_radius = []
        self.list_center = []
        self.list_width_adjustment = []
        self.list_height_adjustment = []

        # TODO stablization module tends to rotate lane polygons, reg, unsub zones...
        if config is not None and video_file in config.map_video_to_configs:
            self.config_lane_slots = config.map_video_to_configs[video_file].polygons
        else:
            self.config_lane_slots = []

        self.list_frequency_to_update_config = []
        self.list_index_to_update_config = []
        self.list_frame_id_to_update_config = []
        self.center = 0, 0
        self.width_starting_point_authentic_resolution_roi = 0
        self.height_starting_point_authentic_resolution_roi = 0
        self.width_authentic_resolution = -1
        self.height_authentic_resolution = -1
        self.width_authentic_resolution_roi = -1
        self.height_authentic_resolution_roi = -1
        self.input_video_width = -1
        self.input_video_height = -1
        self.correction_rot_angle = 0
        self.correction_shift_width = 0
        self.correction_shift_height = 0

    def set_center(self, width_starting_point_authentic_resolution_roi,
                   height_starting_point_authentic_resolution_roi,
                   width_authentic_resolution_roi,
                   height_authentic_resolution_roi,
                   input_video_width,
                   input_video_height,
                   width_authentic_resolution,
                   height_authentic_resolution):

        self.width_starting_point_authentic_resolution_roi = width_starting_point_authentic_resolution_roi
        self.height_starting_point_authentic_resolution_roi = height_starting_point_authentic_resolution_roi
        self.width_authentic_resolution = width_authentic_resolution
        self.height_authentic_resolution = height_authentic_resolution
        self.width_authentic_resolution_roi = width_authentic_resolution_roi
        self.height_authentic_resolution_roi = height_authentic_resolution_roi
        self.input_video_width = input_video_width
        self.input_video_height = input_video_height

        w, h = input_video_width / 2 - width_starting_point_authentic_resolution_roi, \
                input_video_height / 2 - height_starting_point_authentic_resolution_roi

        w *= width_authentic_resolution / width_authentic_resolution_roi
        h *= height_authentic_resolution / height_authentic_resolution_roi
        self.center = w, h

    def set_stabalization_polygons(self, list_polygons):
        """
        this is the main method to establish the N stabliztion zones on the image
        :param list_polygons:
        :return:
        """
        self.list_polygons = list_polygons
        self.list_polygons_original = [p for p in self.list_polygons]
        self.list_polygons_latest = [p for p in self.list_polygons]

        for index_poly, poly in enumerate(list_polygons):
            bounding_box = poly.minimum_rotated_rectangle
            bounding_box_corners = list(zip(*bounding_box.exterior.coords.xy))
            poly_corners = list(zip(*poly.exterior.coords.xy))

            box_radius_short, box_radius_long = np.finfo(np.float).max, -1
            for i in range(len(bounding_box_corners) - 1):
                p, p_next = bounding_box_corners[i], bounding_box_corners[i + 1]
                dist = np.sqrt((p[0] - p_next[0]) ** 2 + (p[1] - p_next[1]) ** 2)
                box_radius_short = min(box_radius_short, dist)
                box_radius_long = max(box_radius_long, dist)
            self.list_box_short_radius.append(box_radius_short)
            self.list_box_long_radius.append(box_radius_long)

            max_w, min_w, max_h, min_h = -1, np.finfo(np.float).max, -1, np.finfo(np.float).max
            for i in range(len(poly_corners)):
                w, h = poly_corners[i]
                max_w = max(max_w, w)
                max_h = max(max_h, h)
                min_w = min(min_w, w)
                min_h = min(min_h, h)

            width = max_w - min_w
            height = max_h - min_h

            # Note: this is easily giving a bug, the way how center is calculated
            self.list_center.append(((max_w + min_w) * 0.5, (max_h + min_h) * 0.5))

            # Note: here list of widths and height is for AABB box, width is horizontal axis and height is veticle axis on image
            self.list_widths.append(width)
            self.list_heights.append(height)
            
            self.list_width_adjustment.append([])
            self.list_height_adjustment.append([])
            self.list_index_to_update_config.append(0)
            self.list_frame_id_to_update_config.append(0)

            if index_poly == 0:
                self.list_frequency_to_update_config.append(1)
            else:
                self.list_frequency_to_update_config.append(1)

    def set_stabalization_list(self, list_frac_coords):
        self.list_frac_pixel_cords = list_frac_coords

    def is_a_hit(self, box_center, bbox_width, bbox_height, frame_id, mandatory=False):
        """
        here 2nd parameter is width, 3rd parameter is length for AABB, width is horozontal axis, height is vertical axis on image

        here all the coordinates are pixels in tracking image space
        """
        if len(self.list_polygons_original) <= 0:
            return None

        adjustment_needed = False

        x_c, y_c = box_center
        point = geometry.Point(x_c, y_c)
        for i in range(len(self.list_polygons_original)):
            # TODO only take the first two stablization zones
            if i >= 2:
                continue

            # use latest polygon to detect hit update
            poly = self.list_polygons_latest[i]
            width = self.list_widths[i]
            height = self.list_heights[i]
            center_w, center_h = self.list_center[i]
            
            if poly.contains(point) or is_overlap(x_c, y_c, bbox_width, bbox_height, center_w, center_h, width, height):
                delta_width = np.abs(width - bbox_width) / width
                delta_height = np.abs(height - bbox_height) / height

                if mandatory is True or delta_width < 0.4 and delta_height < 0.4:

                    height_adjustment = y_c - center_h
                    width_adjustment = x_c - center_w

                    if frame_id == 0:
                        logging.info("stablization index %s x %s y %s w %s h %s %s" % (i, x_c, y_c, center_w, center_h, self.list_center[i]))

                    self.list_width_adjustment[i].append(width_adjustment)
                    self.list_height_adjustment[i].append(height_adjustment)

                    index_to_update = self.list_index_to_update_config[i]

                    if index_to_update % self.list_frequency_to_update_config[i] == 0:

                        points = []
                        old_poly = self.list_polygons_original[i]
                        for j in range(len(old_poly.exterior.coords)):
                            p = old_poly.exterior.coords[j]
                            w, h = p
                            points.append((w + width_adjustment, h + height_adjustment))

                        new_poly = geometry.Polygon(points)

                        # updating the box for the stablization zone
                        self.list_polygons[i] = new_poly
                        self.list_polygons_latest[i] = new_poly

                        self.list_frame_id_to_update_config[i] = frame_id
                        adjustment_needed = True

                    self.list_index_to_update_config[i] += 1
                    break

                else:
                    pass

        if adjustment_needed is False:
            return

        # Note these two coordinates are in tracking space
        center_point_w, center_point_h = self.center

        if len(self.list_width_adjustment) >= 2 and len(self.list_width_adjustment[0]) > 0 and \
                len(self.list_width_adjustment[1]) > 0 and \
                np.abs(self.list_frame_id_to_update_config[0] - self.list_frame_id_to_update_config[1]) < 60:

            w0_ori, h0_ori = self.list_center[0]
            w1_ori, h1_ori = self.list_center[1]

            w0 = w0_ori + (self.list_width_adjustment[0][-1] if len(self.list_width_adjustment[0]) > 0 else 0)
            h0 = h0_ori + (self.list_height_adjustment[0][-1] if len(self.list_height_adjustment[0]) > 0 else 0)
            w1 = w1_ori + (self.list_width_adjustment[1][-1] if len(self.list_width_adjustment[1]) > 0 else 0)
            h1 = h1_ori + (self.list_height_adjustment[1][-1] if len(self.list_height_adjustment[1]) > 0 else 0)

            w0_ori, h0_ori, w1_ori, h1_ori, w0, h0, w1, h1 = w0_ori - center_point_w,\
                                                             h0_ori - center_point_h,\
                                                             w1_ori - center_point_w,\
                                                             h1_ori - center_point_h,\
                                                             w0 - center_point_w,\
                                                             h0 - center_point_h,\
                                                             w1 - center_point_w,\
                                                             h1 - center_point_h

            theta_deg, alpha, beta = self.correction_function(w0_ori, h0_ori, w1_ori, h1_ori, w0, h0, w1, h1)
        else:
            return

        self.correction_rot_angle = theta_deg
        self.correction_shift_width = alpha
        self.correction_shift_height = beta

        for i in range(len(self.config_lane_slots)):
            lane_slot = self.config_lane_slots[i]
            points = []

            token = "Polygon"
            token += ":%s:%s:%s:%s:" % (lane_slot.lane_dir, lane_slot.slot_dir,
                                        int(lane_slot.lane_id), int(lane_slot.slot_id))

            if lane_slot.lane_dir == "c" and lane_slot.slot_dir == "c":
                c_w, c_h = lane_slot.intersection_center_to_be_saved
                n_w, n_h = lane_slot.intersection_north_to_be_saved
                e_w, e_h = lane_slot.intersection_east_to_be_saved

                c_w, c_h = self.rescale(c_w, c_h)
                c_w, c_h = self.apply_correction(c_w, c_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                c_w, c_h = self.rescale_inverse(c_w, c_h)

                n_w, n_h = self.rescale(n_w, n_h)
                n_w, n_h = self.apply_correction(n_w, n_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                n_w, n_h = self.rescale_inverse(n_w, n_h)

                e_w, e_h = self.rescale(e_w, e_h)
                e_w, e_h = self.apply_correction(e_w, e_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                e_w, e_h = self.rescale_inverse(e_w, e_h)

                token += "%s,%s," % (int(c_w), int(c_h))
                token += "%s,%s," % (int(n_w), int(n_h))
                token += "%s,%s," % (int(e_w), int(e_h))

            for j in range(len(lane_slot.polygon_to_be_saved.exterior.coords)):
                # Note: these polygon boundary points are in authentic or original resolution pixel coords
                w, h = lane_slot.polygon_to_be_saved.exterior.coords[j]

                w -= self.width_starting_point_authentic_resolution_roi
                h -= self.height_starting_point_authentic_resolution_roi

                # Note self.width_authentic_resolution and self.height_authentic_resolution are
                # actually tracking space, naming errors......
                w *= self.width_authentic_resolution / self.width_authentic_resolution_roi
                h *= self.height_authentic_resolution / self.height_authentic_resolution_roi

                w, h = self.apply_correction(w, h, center_point_w, center_point_h, theta_deg, alpha, beta)

                # Note: convert w and h from tracking space pixels to original image resolution again....
                w *= self.width_authentic_resolution_roi / self.width_authentic_resolution
                h *= self.height_authentic_resolution_roi / self.height_authentic_resolution

                w += self.width_starting_point_authentic_resolution_roi
                h += self.height_starting_point_authentic_resolution_roi

                if j < len(lane_slot.polygon_to_be_saved.exterior.coords) - 1:
                    token += "%s,%s," % (int(w), int(h))

                points.append((w, h))

            token = token[:-1]
            if mandatory:
                print(token)

            # Note: the polygon point coordinate here is in original image space
            new_poly = geometry.Polygon(points)
            lane_slot.polygon = new_poly

            if lane_slot.lane_dir == "c" and lane_slot.slot_dir == "c":
                lane_slot.intersection_center = int(c_w), int(c_h)
                lane_slot.intersection_north = int(n_w), int(n_h)
                lane_slot.intersection_east = int(e_w), int(e_h)

                for j in range(len(lane_slot.triangles_to_be_saved)):
                    triangle_poly = lane_slot.triangles_to_be_saved[j]

                    list_triangle_points = []
                    for kk in range(len(triangle_poly.exterior.coords)):
                        kk_x, kk_y = triangle_poly.exterior.coords[kk]
                        kk_x, kk_y = self.rescale(kk_x, kk_y)
                        kk_x, kk_y = self.apply_correction(kk_x, kk_y, center_point_w, center_point_h, theta_deg, alpha, beta)
                        kk_x, kk_y = self.rescale_inverse(kk_x, kk_y)
                        list_triangle_points.append((kk_x, kk_y))

                    lane_slot.triangles[j] = geometry.Polygon(list_triangle_points)

                lane_slot.setup_intersection_orientation()

            else:
                lane_slot.setup_directions_of_polygon()

    def initial_lane_polygon_adjustment(self, theta_deg, alpha, beta):
        # Note these two coordinates are in tracking space
        center_point_w, center_point_h = self.center

        for i in range(len(self.config_lane_slots)):
            lane_slot = self.config_lane_slots[i]
            points = []

            token = "Polygon"
            token += ":%s:%s:%s:%s:" % (lane_slot.lane_dir, lane_slot.slot_dir,
                                        int(lane_slot.lane_id), int(lane_slot.slot_id))

            if lane_slot.lane_dir == "c" and lane_slot.slot_dir == "c":
                c_w, c_h = lane_slot.intersection_center_to_be_saved
                n_w, n_h = lane_slot.intersection_north_to_be_saved
                e_w, e_h = lane_slot.intersection_east_to_be_saved

                c_w, c_h = self.rescale(c_w, c_h)
                c_w, c_h = self.apply_correction(c_w, c_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                c_w, c_h = self.rescale_inverse(c_w, c_h)

                n_w, n_h = self.rescale(n_w, n_h)
                n_w, n_h = self.apply_correction(n_w, n_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                n_w, n_h = self.rescale_inverse(n_w, n_h)

                e_w, e_h = self.rescale(e_w, e_h)
                e_w, e_h = self.apply_correction(e_w, e_h, center_point_w, center_point_h, theta_deg, alpha, beta)
                e_w, e_h = self.rescale_inverse(e_w, e_h)

                token += "%s,%s," % (int(c_w), int(c_h))
                token += "%s,%s," % (int(n_w), int(n_h))
                token += "%s,%s," % (int(e_w), int(e_h))

            for j in range(len(lane_slot.polygon_to_be_saved.exterior.coords)):
                # Note: these polygon boundary points are in authentic or original resolution pixel coords
                w, h = lane_slot.polygon_to_be_saved.exterior.coords[j]

                w -= self.width_starting_point_authentic_resolution_roi
                h -= self.height_starting_point_authentic_resolution_roi
                w *= self.width_authentic_resolution / self.width_authentic_resolution_roi
                h *= self.height_authentic_resolution / self.height_authentic_resolution_roi

                w, h = self.apply_correction(w, h, center_point_w, center_point_h, theta_deg, alpha, beta)

                # Note: convert w and h from tracking space pixels to original image resolution again....
                w *= self.width_authentic_resolution_roi / self.width_authentic_resolution
                h *= self.height_authentic_resolution_roi / self.height_authentic_resolution

                w += self.width_starting_point_authentic_resolution_roi
                h += self.height_starting_point_authentic_resolution_roi

                if j < len(lane_slot.polygon_to_be_saved.exterior.coords) - 1:
                    token += "%s,%s," % (int(w), int(h))

                points.append((w, h))

            token = token[:-1]

            # Note: the polygon point coordinate here is in original image space
            new_poly = geometry.Polygon(points)
            lane_slot.polygon = new_poly
            lane_slot.polygon_to_be_saved = new_poly

            if lane_slot.lane_dir == "c" and lane_slot.slot_dir == "c":
                lane_slot.intersection_center = int(c_w), int(c_h)
                lane_slot.intersection_north = int(n_w), int(n_h)
                lane_slot.intersection_east = int(e_w), int(e_h)

                for j in range(len(lane_slot.triangles_to_be_saved)):
                    triangle_poly = lane_slot.triangles_to_be_saved[j]

                    list_triangle_points = []
                    for kk in range(len(triangle_poly.exterior.coords)):
                        kk_x, kk_y = triangle_poly.exterior.coords[kk]
                        kk_x, kk_y = self.rescale(kk_x, kk_y)
                        kk_x, kk_y = self.apply_correction(kk_x, kk_y, center_point_w, center_point_h, theta_deg, alpha, beta)
                        kk_x, kk_y = self.rescale_inverse(kk_x, kk_y)
                        list_triangle_points.append((kk_x, kk_y))

                    lane_slot.triangles[j] = geometry.Polygon(list_triangle_points)

                lane_slot.setup_intersection_orientation()

            else:
                lane_slot.setup_directions_of_polygon()

    def rescale(self, w, h):
        w -= self.width_starting_point_authentic_resolution_roi
        h -= self.height_starting_point_authentic_resolution_roi
        w *= self.width_authentic_resolution / self.width_authentic_resolution_roi
        h *= self.height_authentic_resolution / self.height_authentic_resolution_roi
        return w, h

    def rescale_inverse(self, w, h):
        w *= self.width_authentic_resolution_roi / self.width_authentic_resolution
        h *= self.height_authentic_resolution_roi / self.height_authentic_resolution
        w += self.width_starting_point_authentic_resolution_roi
        h += self.height_starting_point_authentic_resolution_roi
        return w, h

    def apply_correction(self, w, h, center_point_w, center_point_h, theta_deg, alpha, beta):
        """
        here alpha is the shift on width dimension, beta is shift on height dimension in units of pixels in tracking
        space

        theta is the rotation angle in degrees
        :param w:
        :param h:
        :param center_point_w:
        :param center_point_h:
        :param theta_deg:
        :param alpha:
        :param beta:
        :return:
        """
        w -= center_point_w
        h -= center_point_h

        theta = theta_deg / 180 * np.pi

        w = alpha + (np.cos(theta) * w - np.sin(theta) * h)
        h = beta + (np.sin(theta) * w + np.cos(theta) * h)

        w += center_point_w
        h += center_point_h

        return w, h

    def correction_function(self, w0_ori, h0_ori, w1_ori, h1_ori, w0, h0, w1, h1):

        dw_p = w1 - w0
        dh_p = h1 - h0
        dw = w1_ori - w0_ori
        dh = h1_ori - h0_ori

        numerator = dw_p * dw + dh_p * dh
        denominator = dw ** 2 + dh ** 2

        cosine_theta = numerator / denominator

        numerator_sine = dh_p * dw - dw_p * dh
        denominator_sine = dw ** 2 + dh ** 2

        sine_theta = numerator_sine / denominator_sine

        cosine_theta = np.clip(cosine_theta, -1, 1)
        sine_theta = np.clip(sine_theta, -1, 1)

        theta_c = np.arccos(cosine_theta)
        theta_s = np.arcsin(sine_theta)

        theta = theta_s

        theta_deg = theta / np.pi * 180.

        alpha0 = w0 - (np.cos(theta) * w0_ori - np.sin(theta) * h0_ori)
        beta0 = h0 - (np.sin(theta) * w0_ori + np.cos(theta) * h0_ori)

        alpha1 = w1 - (np.cos(theta) * w1_ori - np.sin(theta) * h1_ori)
        beta1 = h1 - (np.sin(theta) * w1_ori + np.cos(theta) * h1_ori)

        alpha = (alpha0 + alpha1) / 2
        beta = (beta0 + beta1) / 2

        return theta_deg, alpha, beta


