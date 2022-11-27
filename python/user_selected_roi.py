import os
import cv2
import logging

try:
    from shapely import geometry
except:
    pass

from constant_values import WINDOW_NAME, BLUE


global_frame = None
global_selected_points = []
global_frac_selected_points = []
global_height, global_width, global_depth = -1, -1, -1
global_window_name = None


def get_user_selected_roi(frame, name, auto_processing=False, window_name=""):
    # Note: in batch processing mode, we do not need interactively select ROI, they are given by config text files
    if auto_processing:
        return [[]]

    global global_frame, global_selected_points, global_height, global_width, global_depth, \
        global_frac_selected_points, global_window_name

    frame = frame.copy()

    global_frame = frame
    global_height, global_width, global_depth = global_frame.shape

    global_selected_points.clear()
    global_frac_selected_points.clear()

    if len(window_name) > 0:
        global_window_name = window_name
    else:
        global_window_name = WINDOW_NAME

    cv2.setMouseCallback(global_window_name, click_and_crop)
    logging.info("user_selected_roi: please point & click to select your ROI of %s, type 'd' key when you are done" %
                 name)

    result = []
    while True:
        cv2.imshow(global_window_name, frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('d'):
            logging.info("user_selected_roi: number of selected points for the ROI of %s: %s. Done!!!!!!" % (name,
                                                                                          len(global_selected_points)))
            result.append([i for i in global_frac_selected_points])
            global_frac_selected_points.clear()
            global_selected_points.clear()
            break
        if key & 0xFF == ord('n'):

            result.append([i for i in global_frac_selected_points])
            global_frac_selected_points.clear()
            global_selected_points.clear()

            logging.info("user_selected_roi: please select another ROI of %s" % name)
            continue

    return result


def click_and_crop(event, x, y, flags, param):
    global global_frame, global_window_name
    global global_selected_points, global_height, global_width, global_depth, global_frac_selected_points

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

        # Note: here x and y are in pixel units of the image shown in CV2 Window
        frac_point = (x / global_width, y / global_height)

        global_selected_points.append(point)
        global_frac_selected_points.append(frac_point)

    elif event == cv2.EVENT_LBUTTONUP:
        len_points = len(global_selected_points)
        if len_points > 1:
            cv2.line(global_frame, global_selected_points[len_points - 1], global_selected_points[len_points - 2],
                     (0, 0, 255), 3)
            cv2.imshow(global_window_name, global_frame)


def plot_roi_zone(list_frac_pixel_cords_roi_registration_zone,
                  width_starting_point_authentic_resolution_roi,
                  height_starting_point_authentic_resolution_roi,
                  input_video_width,
                  input_video_height,
                  output_video_width,
                  output_video_height,
                  frame_output,
                  roi_width,
                  roi_height):
    global global_window_name
    for zone in list_frac_pixel_cords_roi_registration_zone:
        for i in range(len(zone)):
            p = zone[i % len(zone)]
            p_prev = zone[(i - 1) % len(zone)]
            w, h = p
            w, h = w * input_video_width, h * input_video_height
            w, h = w - width_starting_point_authentic_resolution_roi, h - height_starting_point_authentic_resolution_roi
            w *= output_video_width / roi_width
            h *= output_video_height / roi_height
            w, h = int(w), int(h)
            w_prev, h_prev = p_prev
            w_prev, h_prev = w_prev * input_video_width, h_prev * input_video_height
            w_prev, h_prev = w_prev - width_starting_point_authentic_resolution_roi, h_prev - height_starting_point_authentic_resolution_roi
            w_prev *= output_video_width / roi_width
            h_prev *= output_video_height / roi_height
            w_prev, h_prev = int(w_prev), int(h_prev)
            cv2.line(frame_output, (w, h), (w_prev, h_prev),
                     BLUE, 3)
            # Note: why send this frame to the window GUI..... seems not needed at all
            #cv2.imshow(global_window_name, frame_output)


def plot_roi_zone_polygon(list_polygons,
                  output_video_width,
                  output_video_height,
                  frame_output,
                  roi_width,
                  roi_height, color):
    global global_window_name
    for poly in list_polygons:
        zone = poly.exterior.coords
        for i in range(len(zone) - 1):
            p = zone[i]
            p_prev = zone[i + 1]
            w, h = p
            w *= output_video_width / roi_width
            h *= output_video_height / roi_height
            w, h = int(w), int(h)
            w_prev, h_prev = p_prev
            w_prev *= output_video_width / roi_width
            h_prev *= output_video_height / roi_height
            w_prev, h_prev = int(w_prev), int(h_prev)
            cv2.line(frame_output, (w, h), (w_prev, h_prev),
                     color, 3)
            # Note: why send this frame to the window GUI..... seems not needed at all
            # cv2.imshow(global_window_name, frame_output)


def plot_center(center,
                  output_video_width,
                  output_video_height,
                  frame_output,
                  roi_width,
                  roi_height, color):
    w, h = center
    w *= output_video_width / roi_width
    h *= output_video_height / roi_height
    w, h = int(w), int(h)
    cv2.circle(frame_output, (w, h), 16, color, -1)


def plot_center_using_scale(center,
                width_starting_point_roi,
                height_starting_point_roi,
                scale_width,
                scale_height,
                frame_output,
                color):
    w, h = center
    w -= width_starting_point_roi
    h -= height_starting_point_roi
    w *= scale_width
    h *= scale_height
    w, h = int(w), int(h)
    cv2.circle(frame_output, (w, h), 16, color, -1)


def save_user_selected_roi(file_path, base_name, list_frac_pixel_cords_roi, list_frac_pixel_cords_roi_registration,
                           list_frac_pixel_cords_roi_occlusion, list_frac_pixel_cords_roi_unsub,
                           list_frac_pixel_cords_roi_stablization):
    file_name = os.path.join(file_path, base_name)
    with open(file_name, 'w') as f:
        for i in range(len(list_frac_pixel_cords_roi)):
            if i == 0:
                f.write("%.8f,%.8f" % (list_frac_pixel_cords_roi[i][0], list_frac_pixel_cords_roi[i][1]))
            else:
                f.write(",%.8f,%.8f" % (list_frac_pixel_cords_roi[i][0], list_frac_pixel_cords_roi[i][1]))
        f.write("\n")

        if len(list_frac_pixel_cords_roi_registration) > 0 and len(list_frac_pixel_cords_roi_registration[0]) > 0:
            f.write("%d\n" % len(list_frac_pixel_cords_roi_registration))
            for zone in list_frac_pixel_cords_roi_registration:
                for ii in range(len(zone)):
                    if ii == 0:
                        f.write("%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                    else:
                        f.write(",%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                f.write("\n")
        else:
            f.write("0\n")

        if len(list_frac_pixel_cords_roi_occlusion) > 0 and len(list_frac_pixel_cords_roi_occlusion[0]) > 0:
            f.write("%d\n" % len(list_frac_pixel_cords_roi_occlusion))
            for zone in list_frac_pixel_cords_roi_occlusion:
                for ii in range(len(zone)):
                    if ii == 0:
                        f.write("%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                    else:
                        f.write(",%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                f.write("\n")
        else:
            f.write("0\n")

        if len(list_frac_pixel_cords_roi_unsub) > 0 and len(list_frac_pixel_cords_roi_unsub[0]) > 0:
            f.write("%d\n" % len(list_frac_pixel_cords_roi_unsub))
            for zone in list_frac_pixel_cords_roi_unsub:
                for ii in range(len(zone)):
                    if ii == 0:
                        f.write("%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                    else:
                        f.write(",%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                f.write("\n")
        else:
            f.write("0\n")

        if len(list_frac_pixel_cords_roi_stablization) > 0 and len(list_frac_pixel_cords_roi_stablization[0]) > 0:
            f.write("%d\n" % len(list_frac_pixel_cords_roi_stablization))
            for zone in list_frac_pixel_cords_roi_stablization:
                for ii in range(len(zone)):
                    if ii == 0:
                        f.write("%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                    else:
                        f.write(",%.8f,%.8f" % (zone[ii][0], zone[ii][1]))
                f.write("\n")
        else:
            f.write("0\n")


def read_user_selected_roi_file(file_path, base_name):
    list_frac_pixel_cords_roi = []
    list_frac_pixel_cords_roi_registration = []
    list_frac_pixel_cords_roi_occlusion = []
    list_frac_pixel_cords_roi_unsub = []
    list_frac_pixel_cords_roi_stablization = []

    file_name = os.path.join(file_path, base_name)
    if not os.path.exists(file_name):
        logging.warning("Warning: no ROI file found!!!!")
        return list_frac_pixel_cords_roi, list_frac_pixel_cords_roi_registration, list_frac_pixel_cords_roi_occlusion, \
            list_frac_pixel_cords_roi_unsub, list_frac_pixel_cords_roi_stablization

    with open(file_name, 'r') as f:
        line = f.readline()
        line.strip()
        l_p = line.split(',')
        for i in range(int(len(l_p) / 2)):
            list_frac_pixel_cords_roi.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))

        line = f.readline().strip()
        num_zone = int(line) if len(line) > 0 else 0
        zone = []
        for ii in range(num_zone):
            line = f.readline()
            line.strip()
            l_p = line.split(',')
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            list_frac_pixel_cords_roi_registration.append(zone)
            zone = []

        line = f.readline().strip()
        num_zone = int(line) if len(line) > 0 else 0
        zone = []
        for ii in range(num_zone):
            line = f.readline()
            line.strip()
            l_p = line.split(',')
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            list_frac_pixel_cords_roi_occlusion.append(zone)
            zone = []

        line = f.readline().strip()
        num_zone = int(line) if len(line) > 0 else 0
        zone = []
        for ii in range(num_zone):
            line = f.readline()
            line.strip()
            l_p = line.split(',')
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            list_frac_pixel_cords_roi_unsub.append(zone)
            zone = []

        line = f.readline().strip()
        num_zone = int(line) if len(line) > 0 else 0
        zone = []
        for ii in range(num_zone):
            line = f.readline()
            line.strip()
            l_p = line.split(',')
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            list_frac_pixel_cords_roi_stablization.append(zone)
            zone = []

    return list_frac_pixel_cords_roi, list_frac_pixel_cords_roi_registration, list_frac_pixel_cords_roi_occlusion,\
        list_frac_pixel_cords_roi_unsub, list_frac_pixel_cords_roi_stablization


def make_polygons(list_frac_pixel_cords_roi_registration_zone,
                            width_starting_point_authentic_resolution_roi,
                            height_starting_point_authentic_resolution_roi,
                            input_video_width,
                            input_video_height,
                            output_video_width,
                            output_video_height,
                           roi_width,
                           roi_height):
    if len(list_frac_pixel_cords_roi_registration_zone) <= 0:
        return []

    polys = []
    for zone in list_frac_pixel_cords_roi_registration_zone:
        points = []
        for i in range(len(zone)):
            p = zone[i % len(zone)]
            w, h = p
            w, h = w * input_video_width, h * input_video_height
            w, h = w - width_starting_point_authentic_resolution_roi, h - height_starting_point_authentic_resolution_roi
            w *= output_video_width / roi_width
            h *= output_video_height / roi_height
            points.append((w, h))

        poly = geometry.Polygon(points)
        polys.append(poly)
    return polys


def get_road_marks(frame, auto_processing=False):
    """
    the coordinates are fractional pixels, in the original image width x height without applying mask or ROI
    """
    logging.info("user_selected_roi: please select 2 x road marks for stablization of drone movement")
    return get_user_selected_roi(frame, "stablization-zone", auto_processing=auto_processing)


def get_roi(frame, roi_file_path, roi_file_name, auto_processing=False, window_name="", only_select_global_roi=False):

    roi_global = get_user_selected_roi(frame, "global", auto_processing=auto_processing, window_name=window_name)

    assert len(roi_global) == 1, "[ERROR] user_selected_roi: global ROI should only be length-1 list"

    u_list_frac_pixel_cords_roi = roi_global[0]

    if not only_select_global_roi:
        u_list_frac_pixel_cords_roi_registration_zone = get_user_selected_roi(frame, "registration",
                                                                              auto_processing=auto_processing,
                                                                              window_name=window_name)
        u_list_frac_pixel_cords_roi_unsubscription_zone = get_user_selected_roi(frame, "un-subscription",
                                                                                auto_processing=auto_processing,
                                                                                window_name=window_name)

        u_list_frac_pixel_cords_roi_occlusion_zone = get_user_selected_roi(frame, "occlusion",
                                                                                auto_processing=auto_processing,
                                                                                window_name=window_name)

        u_list_frac_pixel_cords_roi_stablization_zone = get_user_selected_roi(frame, "stablization-zone",
                                                                              auto_processing=auto_processing,
                                                                              window_name=window_name)

    # if user didn't point and click and request roi, we check if we have a txt file saved before for roi
    list_frac_pixel_cords_roi, list_frac_pixel_cords_roi_registration_zone, \
        list_frac_pixel_cords_roi_occlusion_zone, \
        list_frac_pixel_cords_roi_unsubscription_zone, \
        list_frac_pixel_cords_roi_stablization_zone = \
        read_user_selected_roi_file(roi_file_path, roi_file_name)

    is_roi_not_selected = len(u_list_frac_pixel_cords_roi) <= 0

    if not only_select_global_roi:
        is_reg_not_selected = len(u_list_frac_pixel_cords_roi_registration_zone) == 1 and \
            len(u_list_frac_pixel_cords_roi_registration_zone[0]) <= 0
        is_occ_not_selected = len(u_list_frac_pixel_cords_roi_occlusion_zone) == 1 and \
            len(u_list_frac_pixel_cords_roi_occlusion_zone[0]) <= 0
        is_unsub_not_selected = len(u_list_frac_pixel_cords_roi_unsubscription_zone) == 1 and \
            len(u_list_frac_pixel_cords_roi_unsubscription_zone[0]) <= 0
        is_sta_not_selected = len(u_list_frac_pixel_cords_roi_stablization_zone) == 1 and \
            len(u_list_frac_pixel_cords_roi_stablization_zone[0]) <= 0
    else:
        is_reg_not_selected = True
        is_occ_not_selected = True
        is_unsub_not_selected = True
        is_sta_not_selected = True

    is_everything_not_selected = is_roi_not_selected and is_reg_not_selected and is_occ_not_selected and \
        is_unsub_not_selected and is_sta_not_selected

    if is_roi_not_selected:
        u_list_frac_pixel_cords_roi = list_frac_pixel_cords_roi

    if is_reg_not_selected:
        u_list_frac_pixel_cords_roi_registration_zone = list_frac_pixel_cords_roi_registration_zone

    if is_occ_not_selected:
        u_list_frac_pixel_cords_roi_occlusion_zone = list_frac_pixel_cords_roi_occlusion_zone

    if is_unsub_not_selected:
        u_list_frac_pixel_cords_roi_unsubscription_zone = list_frac_pixel_cords_roi_unsubscription_zone

    if is_sta_not_selected:
        u_list_frac_pixel_cords_roi_stablization_zone = list_frac_pixel_cords_roi_stablization_zone

    return u_list_frac_pixel_cords_roi, u_list_frac_pixel_cords_roi_registration_zone,\
        u_list_frac_pixel_cords_roi_unsubscription_zone, u_list_frac_pixel_cords_roi_occlusion_zone,\
        u_list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected


def get_roi_using_json(json_dict):
    # Note: default ROI is full image
    u_list_frac_pixel_cords_roi = [(0, 0), (1, 0), (1, 1), (0, 1)]

    u_list_frac_pixel_cords_roi_registration_zone = []
    u_list_frac_pixel_cords_roi_unsubscription_zone = []
    u_list_frac_pixel_cords_roi_occlusion_zone = []
    u_list_frac_pixel_cords_roi_stablization_zone = []
    is_everything_not_selected = True

    if json_dict is None:
        logging.info("user_selected_roi: no calib json object for user ROI....")
        return u_list_frac_pixel_cords_roi, u_list_frac_pixel_cords_roi_registration_zone, \
               u_list_frac_pixel_cords_roi_unsubscription_zone, u_list_frac_pixel_cords_roi_occlusion_zone, \
               u_list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected

    if "ConfigString" not in json_dict:
        logging.error("user_selected_roi: config json invalid!!")
        return u_list_frac_pixel_cords_roi, u_list_frac_pixel_cords_roi_registration_zone, \
               u_list_frac_pixel_cords_roi_unsubscription_zone, u_list_frac_pixel_cords_roi_occlusion_zone, \
               u_list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected

    config_dict = json_dict["ConfigString"]

    if "ImageROI" in config_dict:
        u_list_frac_pixel_cords_roi = []
        l_p = config_dict["ImageROI"].split(",")
        for i in range(int(len(l_p) / 2)):
            u_list_frac_pixel_cords_roi.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))

    if "ImageRegistrationZones" in config_dict:
        num_zone = len(config_dict["ImageRegistrationZones"])
        zone = []
        for ii in range(num_zone):
            line = config_dict["ImageRegistrationZones"][ii]
            l_p = line["ImageZoneBoundary"].split(",")
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            u_list_frac_pixel_cords_roi_registration_zone.append(zone)
            zone = []

    if "ImageUnsubscriptionZones" in config_dict:
        num_zone = len(config_dict["ImageUnsubscriptionZones"])
        zone = []
        for ii in range(num_zone):
            line = config_dict["ImageUnsubscriptionZones"][ii]
            l_p = line["ImageZoneBoundary"].split(",")
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            u_list_frac_pixel_cords_roi_unsubscription_zone.append(zone)
            zone = []

    if "ImageStablizationZones" in config_dict:
        num_zone = len(config_dict["ImageStablizationZones"])
        zone = []
        for ii in range(num_zone):
            line = config_dict["ImageStablizationZones"][ii]
            l_p = line["ImageZoneBoundary"].split(",")
            for i in range(int(len(l_p) / 2)):
                zone.append((float(l_p[i * 2]), float(l_p[i * 2 + 1])))
            u_list_frac_pixel_cords_roi_stablization_zone.append(zone)
            zone = []

    return u_list_frac_pixel_cords_roi, u_list_frac_pixel_cords_roi_registration_zone,\
        u_list_frac_pixel_cords_roi_unsubscription_zone, u_list_frac_pixel_cords_roi_occlusion_zone,\
        u_list_frac_pixel_cords_roi_stablization_zone, is_everything_not_selected
