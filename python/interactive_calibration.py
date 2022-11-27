import cv2
import logging

from constant_values import WINDOW_NAME, WIDTH_WINDOW, HEIGHT_WINDOW

selected_points = []
map_selected_polygons = {}
global_frame = None
global_width_start, global_height_start = 0, 0
global_tracking_space_w, global_tracking_space_h = 0, 0
global_input_w, global_input_h = 0, 0


def interactive_calibration(f, w_start_point, h_start_point, tracking_space_w, tracking_space_h, input_w, input_h):
    """
    here f is the frame after cropping using a ROI
    """
    return calibrate_video_file(f, w_start_point, h_start_point, tracking_space_w, tracking_space_h, input_w, input_h)


def calibrate_video_file(f, w_start_point, h_start_point, tracking_space_w, tracking_space_h, input_w, input_h):
    global global_frame, global_width_start, global_height_start, global_tracking_space_w, global_tracking_space_h,\
        global_input_w, global_input_h

    global_width_start = w_start_point
    global_height_start = h_start_point
    global_frame = f
    global_tracking_space_w = tracking_space_w
    global_tracking_space_h = tracking_space_h
    global_input_w = input_w
    global_input_h = input_h

    cv2.setMouseCallback(WINDOW_NAME, click_and_crop)

    selected_polygons = []
    counter_polygons = 0
    is_to_skip_frame = False

    while True:
        cv2.imshow(WINDOW_NAME, f)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('d') or key & 0xFF == ord('q'):
            logging.info("number of selected polygons: %s. Done!!!!!!" % len(selected_polygons))
            is_to_skip_frame = False
            break
        if key & 0xFF == ord('n'):
            selected_points_copy = [point for point in selected_points]
            selected_points.clear()

            logging.info("number of points for the lane polygon: %s" % len(selected_points_copy))
            center_polygon = plot_selected_polygon(f, selected_points_copy)

            lane_id, slot_id, lane_dir, slot_dir, inter_orientation_list = user_input_lane_id_slot_id(global_frame,
                                                                                                      center_polygon)

            if len(inter_orientation_list) > 0:
                _ = plot_selected_polygon(global_frame, inter_orientation_list)
                for point in inter_orientation_list[::-1]:
                    selected_points_copy.insert(0, point)

            map_selected_polygons[counter_polygons] = [lane_id, slot_id, lane_dir, slot_dir, selected_points_copy]
            selected_polygons.append(selected_points_copy)
            selected_points.clear()
            counter_polygons += 1
            continue
        if key & 0xFF == ord('a'):
            logging.info("clear selection of points!!!")
            selected_points.clear()
            continue
        if key & 0xFF == ord('s'):
            logging.info("skip frame!")
            selected_points.clear()
            is_to_skip_frame = True
            break

    save_calibration_results(map_selected_polygons)
    return is_to_skip_frame


def click_and_crop(event, x, y, flags, param):
    global global_frame, global_width_start, global_height_start
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_authen = (x + global_width_start, y + global_height_start)

        logging.info("selected point [ (x, y) original resolution / (x, y) tracking resolution / "
              "(f_x. f_y) fractional original resolution]: ",
              point_authen,
              (x * global_tracking_space_w / global_frame.shape[1],
               y * global_tracking_space_h / global_frame.shape[0]),
              ((x + global_width_start) / global_input_w, (y + global_height_start) / global_input_h))

        selected_points.append(point)

    elif event == cv2.EVENT_LBUTTONUP:
        if len(selected_points) >= 2:
            cv2.line(global_frame, selected_points[len(selected_points) - 1], selected_points[len(selected_points) - 2],
                     (0, 0, 255), 1)
            cv2.imshow(WINDOW_NAME, global_frame)


def user_input_lane_id_slot_id(image, center_polygon):

    center_x, centen_y = center_polygon
    lane_id, slot_id, lane_dir, slot_dir = None, None, None, None
    intersection_id = None
    intersection_orientation_list = []

    logging.info("please select a point to define lane polygon direction along laneID axis "
        "(low to high lane ID)")
    logging.info("please select a point to define lane polygon direction along slotID axis "
        "(low to high slot ID)")

    while True:
        key = cv2.waitKey(0)

        if key & 0xFF == ord('n'):
            if len(selected_points) == 2:
                x_convert = int(selected_points[0][0] + global_width_start)
                y_convert = int(selected_points[0][1] + global_height_start)
                lane_dir = "%s,%s" % (int(x_convert), int(y_convert))
                x_convert = int(selected_points[1][0] + global_width_start)
                y_convert = int(selected_points[1][1] + global_height_start)
                slot_dir = "%s,%s" % (int(x_convert), int(y_convert))
                break
            elif len(selected_points) == 0:
                lane_dir = 'c'
                slot_dir = 'c'
                break
            else:
                logging.warning("wrong number of points selected, try again!")
                selected_points.clear()
                continue

    while True:
        try:
            text = input("[INFO] please type lane_id:")
            lane_id = int(text)
            text = input("[INFO] pleas type slot_id:")
            slot_id = int(text)

            if lane_dir == 'c' and slot_dir == 'c':
                lane_id = 100
                slot_id = 100
                text = input("[INFO] please type intersection_id:")
                intersection_id = int(text)
                lane_id = intersection_id
                slot_id = intersection_id

                logging.info("please point & click intersection center, east bound point, north bound point:")
                selected_points.clear()

                while True:
                    key = cv2.waitKey(0)

                    if key & 0xFF == ord('n'):
                        assert len(selected_points) >= 3,\
                            "you have to select exactly three points to determine intersection orientation! " \
                            "but there are %s points" % (len(selected_points))
                        intersection_orientation_list = [point for point in selected_points[0:3]]

                        break

            break
        except ValueError:
            logging.warning("wrong lane_id!!! try again!!!")

    assert lane_id is not None and slot_id is not None, "wrong lane_id and slot_id!!!!"
    assert lane_dir is not None and slot_dir is not None, "wrong lane_dir and slot_dir!!!!"

    if intersection_id is None:
        cv2.putText(image, "lane %s slot %s" % (lane_id, slot_id),
                    (center_x, centen_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
        lane_dir_point = lane_dir.split(',')
        slot_dir_point = slot_dir.split(',')
        cv2.circle(image, (int(int(lane_dir_point[0]) - global_width_start),
                           int(int(lane_dir_point[1]) - global_height_start)),
                   10, (255, 0, 0), -1)
        cv2.circle(image, (int(int(slot_dir_point[0]) - global_width_start),
                           int(int(slot_dir_point[1]) - global_height_start)),
                   10, (0, 255, 0), -1)
    else:
        cv2.putText(image, "intersection %s (%s %s)" % (intersection_id, lane_dir, slot_dir),
                    (center_x, centen_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

    return lane_id, slot_id, lane_dir, slot_dir,  intersection_orientation_list


def save_calibration_results(map_polygons, ratio_width_raw_display=1, ratio_height_raw_display=1):
    logging.info("Polygons:%s" % len(map_polygons))
    for key, value in map_polygons.items():
        lane_id, slot_id, lane_dir, slot_dir, points = value

        converted_points = []
        for point in points:
            x, y = point
            x_convert = int(x + global_width_start)
            y_convert = int(y + global_height_start)
            converted_points.append((x_convert, y_convert))

        token = "Polygon:%s:%s:%s:%s:" % (lane_dir, slot_dir, lane_id, slot_id)
        for point in converted_points:
            token += "%s,%s," % (point[0], point[1])
        token = token[:-1]
        logging.info(token)


def plot_selected_polygon(image, points):
    center_x, center_y = 0, 0
    for index in range(len(points) - 1):
        point1 = points[index]
        point2 = points[index + 1]
        center_x += point1[0]
        center_y += point1[1]

        cv2.line(image, point1, point2, (255, 0, 0), 1)

    center_x += points[len(points) - 1][0]
    center_y += points[len(points) - 1][1]
    cv2.line(image, points[len(points) - 1], points[0], (255, 0, 0), 1)
    return int(center_x / len(points)), int(center_y / len(points))


