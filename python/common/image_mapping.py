import sys
import cv2
import numpy as np
import logging

import geopy.distance

sys.path.insert(0, '..')
from constant_values import WARP_ALPHA, WARP_BETA, WARP_GAMMA


class ImageMapping(object):
    """
    direct and reverse mapping of image, x -> y ' -> x '' euler rotations, plus a shift defined by x0, y0, z0
    reference paper: Xing, et al.
    https://portal.slac.stanford.edu/sites/lcls_public/instruments/mec/Documents_xrd/XRD_paper_RSI.pdf

    Note: the warp function provided by OpenCV does not take into account of the Jacobian matrix, thus does not
    conserve total number of photon counts
    """
    def __init__(self, alpha, beta, gamma, x0, y0, z0, input_video_width, input_video_height,
                 z_ref=50.1, pixel_size_at_projection_plane=0.1):
        """
        here the angles are in units of degrees for human readable tweaking of camera position and orientation
        :param alpha:
        :param beta:
        :param gamma:
        :param x0:
        :param y0:
        :param z0:
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.z_ref = z_ref
        self.pixel_size_at_projection_plane = pixel_size_at_projection_plane

        self.rotation_matrix = None
        self.shift_matrix = None

        # Note these are just some typical sensor parameters
        self.sensor_size_width = 3.5e-3  # 3.5 mm
        self.sensor_size_height = 4.8e-3  # 4.8 mm

        self.num_pixels_width = 3024
        self.num_pixels_height = 4032

        self.pixel_size_width = self.sensor_size_width / self.num_pixels_width
        self.pixel_size_height = self.sensor_size_height / self.num_pixels_height

        self.build_warp_transform()

        # Note this is for the original image
        self.max_x = np.finfo(np.float64).min
        self.min_x = np.finfo(np.float64).max
        self.max_y = np.finfo(np.float64).min
        self.min_y = np.finfo(np.float64).max
        self.M, self.image_width_proj, self.image_height_proj = None, None, None

        # Note this is for the output image after cropping, resizing....
        self.max_x_output = np.finfo(np.float64).min
        self.min_x_output = np.finfo(np.float64).max
        self.max_y_output = np.finfo(np.float64).min
        self.min_y_output = np.finfo(np.float64).max
        self.M_output, self.image_width_proj_output, self.image_height_proj_output = None, None, None

        self.get_transform_matrix_for_open_cv_warp_perspective_use_image_dimension(input_video_width,
                                                                                   input_video_height)
        logging.debug("image_mapping: the warped image: width %s x height %s" %
              (self.image_width_proj, self.image_height_proj))

        self.calib_parameter_list = None
        self.is_image_to_dist_calib_built = False

    def build_warp_transform(self):
        alpha, beta, gamma = self.alpha / 180. * np.pi, self.beta / 180. * np.pi, self.gamma / 180. * np.pi
        c1 = np.cos(alpha)
        c2 = np.cos(beta)
        c3 = np.cos(gamma)

        s1 = np.sin(alpha)
        s2 = np.sin(beta)
        s3 = np.sin(gamma)

        self.rotation_matrix = np.array([
            [c2,        s2 * s3,                    c3 * s2],
            [s1 * s2,   c1 * c3 - c2 * s1 * s3,     -c1 * s3 - c2 * c3 * s1],
            [-c1 * s2,  c3 * s1 + c1 * c2 * s3,     c1 * c2 * c3 - s1 * s3]
        ]).reshape((3, 3))

        self.shift_matrix = np.array([
            self.x0,
            self.y0,
            self.z0
        ]).reshape((3, 1))

    def projection_along_z_axis(self, pos, z_ref):
        x, y, z = pos[0][0], pos[1][0], pos[2][0]

        # Note: trick for dividing by zero issue
        z += 1e-9

        x_proj = x * z_ref / z
        y_proj = y * z_ref / z
        z_proj = z_ref
        return np.array([x_proj, y_proj, z_proj]).reshape(3, 1)

    def get_pixel_coordinates_at_projected_plane_output(self, pos1, pos2, pos3, pos4, pixel_size):
        self.update_x_y_in_projected_plane_output(pos1)
        self.update_x_y_in_projected_plane_output(pos2)
        self.update_x_y_in_projected_plane_output(pos3)
        self.update_x_y_in_projected_plane_output(pos4)

        num_pixel_x = int((self.max_x_output - self.min_x_output) / pixel_size)
        num_pixel_y = int((self.max_y_output - self.min_y_output) / pixel_size)

        pos1_pixel = self.get_pixel_output(pos1, pixel_size)
        pos2_pixel = self.get_pixel_output(pos2, pixel_size)
        pos3_pixel = self.get_pixel_output(pos3, pixel_size)
        pos4_pixel = self.get_pixel_output(pos4, pixel_size)

        return pos1_pixel, pos2_pixel, pos3_pixel, pos4_pixel, num_pixel_x, num_pixel_y

    def get_pixel_coordinates_at_projected_plane(self, pos1, pos2, pos3, pos4, pixel_size):
        self.update_x_y_in_projected_plane(pos1)
        self.update_x_y_in_projected_plane(pos2)
        self.update_x_y_in_projected_plane(pos3)
        self.update_x_y_in_projected_plane(pos4)

        num_pixel_x = int((self.max_x - self.min_x) / pixel_size)
        num_pixel_y = int((self.max_y - self.min_y) / pixel_size)

        pos1_pixel = self.get_pixel(pos1, pixel_size)
        pos2_pixel = self.get_pixel(pos2, pixel_size)
        pos3_pixel = self.get_pixel(pos3, pixel_size)
        pos4_pixel = self.get_pixel(pos4, pixel_size)

        return pos1_pixel, pos2_pixel, pos3_pixel, pos4_pixel, num_pixel_x, num_pixel_y

    def update_x_y_in_projected_plane(self, pos):
        x_in_meter = pos[0][0]
        y_in_meter = pos[1][0]
        self.max_x = max(x_in_meter, self.max_x)
        self.min_x = min(x_in_meter, self.min_x)
        self.max_y = max(y_in_meter, self.max_y)
        self.min_y = min(y_in_meter, self.min_y)

    def update_x_y_in_projected_plane_output(self, pos):
        x_in_meter = pos[0][0]
        y_in_meter = pos[1][0]
        self.max_x_output = max(x_in_meter, self.max_x_output)
        self.min_x_output = min(x_in_meter, self.min_x_output)
        self.max_y_output = max(y_in_meter, self.max_y_output)
        self.min_y_output = min(y_in_meter, self.min_y_output)

    def get_pixel(self, pos, pixel_size):
        x_in_meter = pos[0][0]
        y_in_meter = pos[1][0]
        return [int((x_in_meter - self.min_x) / pixel_size), int((y_in_meter - self.min_y) / pixel_size)]

    def get_pixel_output(self, pos, pixel_size):
        x_in_meter = pos[0][0]
        y_in_meter = pos[1][0]
        return [int((x_in_meter - self.min_x_output) / pixel_size), int((y_in_meter - self.min_y_output) / pixel_size)]

    def get_transform_matrix_for_open_cv_warp_perspective_use_image_dimension(self, width, height):

        if self.alpha <= 0 and self.beta <= 0 and self.gamma <= 0 and self.x0 <= 0 and self.y0 <= 0 and self.z0 <= 0:
            self.M = np.identity(3)
            self.image_width_proj = width
            self.image_height_proj = height
            return

        point_top_left = np.array([0, 0, 0]).reshape((3, 1))
        point_top_right = np.array([(width - 1) * self.pixel_size_width, 0, 0]).reshape((3, 1))
        point_bottom_right = np.array([(width - 1) * self.pixel_size_width, (height - 1) *
                                       self.pixel_size_height, 0]).reshape((3, 1))
        point_bottom_left = np.array([0, (height - 1) * self.pixel_size_height, 0]).reshape((3, 1))

        # Note: these 4 coordinates below are in units of meters
        point_top_left_global = self.rotation_matrix.dot(point_top_left) + self.shift_matrix
        point_top_right_global = self.rotation_matrix.dot(point_top_right) + self.shift_matrix
        point_bottom_right_global = self.rotation_matrix.dot(point_bottom_right) + self.shift_matrix
        point_bottom_left_global = self.rotation_matrix.dot(point_bottom_left) + self.shift_matrix

        top_left = [0, 0]
        top_right = [width - 1, 0]
        bottom_right = [width - 1, height - 1]
        bottom_left = [0, height - 1]
        pts = np.array([bottom_left, bottom_right, top_right, top_left])

        z_plane_destination = self.z_ref
        pixel_size_at_projected_plane = self.pixel_size_at_projection_plane

        # Note: these 4 coordinates below in units of meters, projected to a downstream z plane
        point_top_left_global_proj = self.projection_along_z_axis(point_top_left_global, z_plane_destination)
        point_top_right_global_proj = self.projection_along_z_axis(point_top_right_global, z_plane_destination)
        point_bottom_right_global_proj = self.projection_along_z_axis(point_bottom_right_global, z_plane_destination)
        point_bottom_left_global_proj = self.projection_along_z_axis(point_bottom_left_global, z_plane_destination)

        top_left_dest, top_right_dest, bottom_right_dest, bottom_left_dest, image_width_proj, image_height_proj = \
            self.get_pixel_coordinates_at_projected_plane(point_top_left_global_proj,
                                                          point_top_right_global_proj,
                                                          point_bottom_right_global_proj,
                                                          point_bottom_left_global_proj,
                                                        pixel_size_at_projected_plane)

        pts_dest = np.array([bottom_left_dest, bottom_right_dest, top_right_dest, top_left_dest])

        pts = np.float32(pts.tolist())
        pts_dest = np.float32(pts_dest.tolist())
        self.M = cv2.getPerspectiveTransform(pts, pts_dest)
        self.image_width_proj = image_width_proj
        self.image_height_proj = image_height_proj

    def get_transform_matrix_for_open_cv_warp_perspective(self, img):

        height, width, depth = img.shape

        if self.alpha <= 0 and self.beta <= 0 and self.gamma <= 0 and self.x0 <= 0 and self.y0 <= 0 and self.z0 <= 0:
            return np.identity(3), width, height

        point_top_left = np.array([0, 0, 0]).reshape((3, 1))
        point_top_right = np.array([(width - 1) * self.pixel_size_width, 0, 0]).reshape((3, 1))
        point_bottom_right = np.array([(width - 1) * self.pixel_size_width, (height - 1) *
                                       self.pixel_size_height, 0]).reshape((3, 1))
        point_bottom_left = np.array([0, (height - 1) * self.pixel_size_height, 0]).reshape((3, 1))

        # Note: these 4 coordinates below are in units of meters
        point_top_left_global = self.rotation_matrix.dot(point_top_left) + self.shift_matrix
        point_top_right_global = self.rotation_matrix.dot(point_top_right) + self.shift_matrix
        point_bottom_right_global = self.rotation_matrix.dot(point_bottom_right) + self.shift_matrix
        point_bottom_left_global = self.rotation_matrix.dot(point_bottom_left) + self.shift_matrix

        top_left = [0, 0]
        top_right = [width - 1, 0]
        bottom_right = [width - 1, height - 1]
        bottom_left = [0, height - 1]
        pts = np.array([bottom_left, bottom_right, top_right, top_left])

        z_plane_destination = self.z_ref
        pixel_size_at_projected_plane = self.pixel_size_at_projection_plane

        # Note: these 4 coordinates below in units of meters, projected to a downstream z plane
        point_top_left_global_proj = self.projection_along_z_axis(point_top_left_global, z_plane_destination)
        point_top_right_global_proj = self.projection_along_z_axis(point_top_right_global, z_plane_destination)
        point_bottom_right_global_proj = self.projection_along_z_axis(point_bottom_right_global, z_plane_destination)
        point_bottom_left_global_proj = self.projection_along_z_axis(point_bottom_left_global, z_plane_destination)

        top_left_dest, top_right_dest, bottom_right_dest, bottom_left_dest, image_width_proj, image_height_proj = \
            self.get_pixel_coordinates_at_projected_plane_output(point_top_left_global_proj,
                                                          point_top_right_global_proj,
                                                          point_bottom_right_global_proj,
                                                          point_bottom_left_global_proj,
                                                        pixel_size_at_projected_plane)

        pts_dest = np.array([bottom_left_dest, bottom_right_dest, top_right_dest, top_left_dest])

        pts = np.float32(pts.tolist())
        pts_dest = np.float32(pts_dest.tolist())
        M = cv2.getPerspectiveTransform(pts, pts_dest)

        return M, image_width_proj, image_height_proj

    def transform(self, frame):
        """
        this is for transforming the output image
        :param frame:
        :return:
        """
        if self.M_output is None:
            self.M_output, self.image_width_proj_output, self.image_height_proj_output =\
                self.get_transform_matrix_for_open_cv_warp_perspective(frame)
        return cv2.warpPerspective(frame, self.M_output, dsize=(self.image_width_proj_output,
                                                                self.image_height_proj_output), flags=cv2.INTER_LINEAR)

    def warp_pixel_coordinates(self, x, y):
        """
        both input and output can be fractional of pixels
        """
        m11, m12, m13 = self.M[0]
        m21, m22, m23 = self.M[1]
        m31, m32, m33 = self.M[2]

        return (m11 * x + m12 * y + m13) / (m31 * x + m32 * y + m33),\
            (m21 * y + m22 * y + m23) / (m31 * x + m32 * y + m33)

    def build_image_to_distance_calibrations(self, video_file, config):
        factor = 1.
        origin1, origin2 = None, None
        factor_dual = None

        video_config = config.map_video_to_configs[video_file]

        for i in range(len(video_config.calibration_point_coordinates) - 1):
            calib_point_a = video_config.calibration_point_coordinates[i]
            calib_point_b = video_config.calibration_point_coordinates[i + 1]

            # these are in pixel coordinates in the original image resolution and no ROI cropping
            x_pixel_a, y_pixel_a = calib_point_a.pixel
            x_pixel_a, y_pixel_a = self.warp_pixel_coordinates(x_pixel_a, y_pixel_a)

            latitude_a, longitude_a = calib_point_a.gps_location

            if calib_point_a.pixel_dual is not None:
                x_pixel_a_dual, y_pixel_a_dual = calib_point_a.pixel_dual

            x_pixel_b, y_pixel_b = calib_point_b.pixel
            x_pixel_b, y_pixel_b = self.warp_pixel_coordinates(x_pixel_b, y_pixel_b)

            latitude_b, longitude_b = calib_point_b.gps_location

            if calib_point_b.pixel_dual is not None:
                x_pixel_b_dual, y_pixel_b_dual = calib_point_b.pixel_dual

            distance_meter = geopy.distance.vincenty((latitude_a, longitude_a), (latitude_b, longitude_b)).km * 1000.
            logging.debug("image_mapping: distance between two consecutive calib. points %s [m]" % distance_meter)

            distance_pixels = np.sqrt((x_pixel_a - x_pixel_b) ** 2 + (y_pixel_a - y_pixel_b) ** 2)
            logging.debug("image_mapping: distance between two consecutive calib. points %s [pixels]" %
                  distance_pixels)

            factor = distance_meter / distance_pixels
            logging.debug("image_mapping: calib. factor: %s [m / pixel]" % factor)

            if calib_point_a.pixel_dual is not None and calib_point_b.pixel_dual is not None:
                distance_pixels_dual = np.sqrt(
                    (x_pixel_a_dual - x_pixel_b_dual) ** 2 + (y_pixel_a_dual - y_pixel_b_dual) ** 2)
                logging.debug("Calibration pixels [dual]: %s" % distance_pixels_dual)
                factor_dual = distance_meter / distance_pixels_dual
                logging.debug("Calibration factor [dual]: %s" % factor_dual)

            origin1 = geopy.Point(latitude_a, longitude_a)
            origin2 = geopy.Point(latitude_b, longitude_b)

        bearing1 = self.get_bearing(latitude_a / 180. * np.pi, longitude_a / 180. * np.pi,
                               latitude_b / 180. * np.pi, longitude_b / 180. * np.pi)

        bearing2 = self.get_bearing(latitude_b / 180. * np.pi, longitude_b / 180. * np.pi,
                               latitude_a / 180. * np.pi, longitude_a / 180. * np.pi)

        logging.debug("image_mapping: bearing angle: %s [deg] from point A to point B" % bearing1)
        logging.debug("image_mapping: bearing angle: %s [deg] from point B to point A" % bearing2)

        x_b_pixel_offset = x_pixel_b - x_pixel_a
        y_b_pixel_offset = y_pixel_b - y_pixel_a

        phi_b_a = np.arctan2(y_b_pixel_offset, x_b_pixel_offset)
        phi_a_b = np.arctan2(-y_b_pixel_offset, -x_b_pixel_offset)

        if calib_point_a.pixel_dual is not None and calib_point_b.pixel_dual is not None:
            x_b_pixel_offset_dual = x_pixel_b_dual - x_pixel_a_dual
            y_b_pixel_offset_dual = y_pixel_b_dual - y_pixel_a_dual
            phi_b_dual = np.arctan2(y_b_pixel_offset_dual, x_b_pixel_offset_dual)

        if factor_dual is None:
            self.calib_parameter_list = factor, (origin1, origin2), (bearing1, bearing2), (x_pixel_a, y_pixel_a),\
                                        (latitude_a, longitude_a), (x_pixel_b, y_pixel_b), (latitude_b, longitude_b),\
                                        (phi_b_a, phi_a_b)
        else:
            self.calib_parameter_list = factor, factor_dual, (origin1, origin2), (bearing1, bearing2),\
                                        (x_pixel_a, y_pixel_a), (x_pixel_a_dual, y_pixel_a_dual),\
                                        (latitude_a, longitude_a), (x_pixel_b, y_pixel_b),\
                                        (x_pixel_b_dual, y_pixel_b_dual), (latitude_b, longitude_b),\
                                        (phi_b_a, phi_a_b), phi_b_dual

        self.is_image_to_dist_calib_built = True

    def get_bearing(self, lat1, lon1, lat2, lon2):
        """
        bearing angle: vector goes from point 1 to point 2, rotating counter-clock wise, in units of degrees, the angle
        between this vector and east
        :param lat1:
        :param lon1:
        :param lat2:
        :param lon2:
        :return:
        """
        dLon = lon2 - lon1
        y = np.sin(dLon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
        bearing = np.rad2deg(np.arctan2(y, x))
        if bearing < 0:
            bearing += 360
        return bearing


def main():
    # Note y' -> x" tilting are most common for CCTV cameras, a negative shift in x axis to move camera to aim its
    # center at the objects that are being monitored
    image_mapping = ImageMapping(0, 45, 0, 0, 0, 0.01, 1920, 1080, z_ref=0.03, pixel_size_at_projection_plane=0.00001)

    image_file = "calibration_image.jpg"

    window_name = "image_mapping"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, 1920, 1080)
    frame = cv2.imread(image_file)

    warped = image_mapping.transform(frame)

    cv2.imshow(window_name, warped)

    while True:
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()
