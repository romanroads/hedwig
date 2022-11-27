import cv2
import numpy as np

try:
    import torch
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode, GenericMask, _create_text_labels
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
except:
    pass
from constant_values import RED, BLUE, YELLOW, YELLOW_RGB, RED_RGB, GREEN

from user_selected_roi import plot_center_using_scale

offset_width = 50
offset_height = 50


def plot_frame_id(image, frame_number, font_scale=1.0, line_thickness=2):
    cv2.putText(image, "Frame %s" % frame_number,
                (offset_width, offset_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                RED,
                line_thickness)


def plot_shapely_polygon(frame, list_of_ground_truth_poly, color=YELLOW):
    for polygon in list_of_ground_truth_poly:
        plot_selected_polygon(frame, list(zip(*polygon.exterior.coords.xy)), color)


def create_text_labels(classes, scores, class_names):
    labels = None

    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]

    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(la, s * 100) for la, s in zip(labels, scores)]
    return labels


def plot_perception_results(frame_output, car_metadata, predictions, scale_width, scale_height, plot_mask, plot_box,
                            plot_polygon, plot_class, matched_index_of_detected_agents, raw_detected_polygons,
                            default_color_code=(1, 1, 1)):
    """
    different plotting modes for boxes (axis-aligned, AABB), polygon boundary, masks, classification results, etc.
    :param frame_output:
    :param car_metadata:
    :param predictions:
    :param scale_width:
    :param scale_height:
    :param plot_mask:
    :param plot_box:
    :param plot_polygon:
    :param plot_class:
    :param matched_index_of_detected_agents:
    :param raw_detected_polygons:
    :param default_color_code:
    :return:
    """
    default_color_code_full_scale = [int(v * 255) for v in default_color_code]
    list_of_mask_points = []

    if plot_mask and predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        for mask in masks:
            seg_map = SegmentationMapsOnImage(mask, shape=mask.shape)
            ori_height, ori_width = mask.shape[0], mask.shape[1]
            seg_map = seg_map.resize((int(scale_height * ori_height), int(scale_width * ori_width)))
            scaled_mask = seg_map.get_arr()

            mask_indices = np.argwhere(scaled_mask == True).astype(np.float)
            mask_indices = mask_indices.astype(np.int)

            for point in mask_indices:
                list_of_mask_points.append((point[1], point[0]))

        masks = None
    else:
        masks = None

    if plot_box or plot_class:
        # Note: plot box has to be ON for classification to printed out as well
        plot_box = True

        v = Visualizer(frame_output[:, :, ::-1],
                       metadata=car_metadata,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION  # remove the colors of unsegmented pixels
                       )

        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        scores = predictions.scores if predictions.has("scores") else None
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if plot_class:
            labels = create_text_labels(classes, scores, car_metadata.get("thing_classes", None))
        else:
            labels = None

        colors = None
        alpha = 0.3

        if plot_box:
            original_tensor = boxes.tensor
            scale_tensor = torch.ones(boxes.tensor.shape)
            scale_tensor[:, 0] *= scale_width
            scale_tensor[:, 1] *= scale_height
            scale_tensor[:, 2] *= scale_width
            scale_tensor[:, 3] *= scale_height

            resized_tensor = torch.mul(boxes.tensor, scale_tensor)
            boxes.tensor = resized_tensor

            colors = []
            for i in range(len(classes)):
                if i in matched_index_of_detected_agents:
                    agent = matched_index_of_detected_agents[i]
                    color_code = [v / 255 for v in agent.color_code]
                    colors.append(color_code)
                else:
                    colors.append(default_color_code)
        else:
            boxes = None

        # Note: when asked to plot box and classes, we do not use detectron API to plot feature points since
        # we plot them in our own tracklet system
        keypoints = None

        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        if plot_box:
            boxes.tensor = original_tensor

        frame_output = v.output.get_image()

    if plot_mask:
        for _point in list_of_mask_points:
            cv2.circle(frame_output, tuple(_point), 1, (255, 0, 0), -1)

    if plot_polygon:
        for index_raw_poly in range(len(raw_detected_polygons)):
            raw_poly = raw_detected_polygons[index_raw_poly]
            if raw_poly is None:
                continue

            corner_1, corner_2, corner_3, corner_4 = raw_poly
            x_corner_1, y_corner_1 = corner_1
            x_corner_2, y_corner_2 = corner_2
            x_corner_3, y_corner_3 = corner_3
            x_corner_4, y_corner_4 = corner_4
            x_corner_1 = int(x_corner_1 * scale_width)
            x_corner_2 = int(x_corner_2 * scale_width)
            x_corner_3 = int(x_corner_3 * scale_width)
            x_corner_4 = int(x_corner_4 * scale_width)
            y_corner_1 = int(y_corner_1 * scale_height)
            y_corner_2 = int(y_corner_2 * scale_height)
            y_corner_3 = int(y_corner_3 * scale_height)
            y_corner_4 = int(y_corner_4 * scale_height)

            color_code = default_color_code_full_scale
            if index_raw_poly in matched_index_of_detected_agents:
                tracklet = matched_index_of_detected_agents[index_raw_poly]
                color_code = tracklet.color_code

            cv2.line(frame_output, (x_corner_1, y_corner_1), (x_corner_2, y_corner_2), color_code, 4)
            cv2.line(frame_output, (x_corner_2, y_corner_2), (x_corner_3, y_corner_3), color_code, 4)
            cv2.line(frame_output, (x_corner_3, y_corner_3), (x_corner_4, y_corner_4), color_code, 4)
            cv2.line(frame_output, (x_corner_4, y_corner_4), (x_corner_1, y_corner_1), color_code, 4)

            cv2.putText(frame_output, "det_%s" % index_raw_poly,
                        (x_corner_1, y_corner_1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        RED,
                        2)

    return frame_output


def plot_selected_polygon(image, points, color):

    integer_list = [(int(point[0]), int(point[1])) for point in points]

    for index in range(len(integer_list) - 1):
        point1 = integer_list[index]
        point2 = integer_list[index + 1]

        cv2.line(image, point1, point2, color, 3)

    cv2.line(image, integer_list[len(points) - 1], integer_list[0], color, 3)


def plot_lane_slot_polygons(image, config, video_file, frame_id, width_starting_point_roi, height_starting_point_roi,
                           scale_width, scale_height):
    video_config = config.map_video_to_configs[video_file]
    list_of_points = []
    list_of_points_updated = []
    color_codes = []

    for polygon in video_config.polygons:
        ts_start, ts_end = polygon.ts_start, polygon.ts_end

        if ts_start <= frame_id < ts_end:
            lane_id, slot_id, poly, poly_updated, lane_dir, slot_dir = polygon.lane_id, polygon.slot_id,\
                polygon.polygon_to_be_saved, polygon.polygon, polygon.lane_dir, polygon.slot_dir

            points = []
            for x, y in poly.exterior.coords:
                points.append((int((x - width_starting_point_roi) * scale_width),
                               int((y - height_starting_point_roi) * scale_height)))
            list_of_points.append(points)
            color_codes.append(polygon.color_code)

            if polygon.triangles is not None:
                for triangle in polygon.triangles:
                    list_triangle_points = [(int((x - width_starting_point_roi) * scale_width),
                                 int((y - height_starting_point_roi) * scale_height)) for x, y in
                                triangle.exterior.coords]
                    list_of_points_updated.append(list_triangle_points)

                list_of_points_updated.append([
                    rescale(polygon.intersection_center, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height),
                    rescale(polygon.intersection_east, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height),
                    rescale(polygon.intersection_north, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height)
                ])

                for triangle in polygon.triangles_to_be_saved:
                    list_triangle_points = [(int((x - width_starting_point_roi) * scale_width),
                                 int((y - height_starting_point_roi) * scale_height)) for x, y in
                                triangle.exterior.coords]
                    list_of_points.append(list_triangle_points)
                    color_codes.append(polygon.color_code)

                list_of_points.append([
                    rescale(polygon.intersection_center_to_be_saved, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height),
                    rescale(polygon.intersection_east_to_be_saved, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height),
                    rescale(polygon.intersection_north_to_be_saved, width_starting_point_roi, height_starting_point_roi,
                            scale_width, scale_height)
                ])
                color_codes.append(polygon.color_code)

            points = []
            for x, y in poly_updated.exterior.coords:
                points.append((int((x - width_starting_point_roi) * scale_width),
                               int((y - height_starting_point_roi) * scale_height)))
            list_of_points_updated.append(points)

            if polygon.north_axis is not None:
                plot_center_using_scale(polygon.intersection_center,
                                        width_starting_point_roi,
                                        height_starting_point_roi,
                                      scale_width,
                                      scale_height,
                                      image,
                                      BLUE)
                plot_center_using_scale(polygon.intersection_east,
                                        width_starting_point_roi,
                                        height_starting_point_roi,
                                        scale_width,
                                        scale_height,
                                        image,
                                        BLUE)
                plot_center_using_scale(polygon.intersection_north,
                                        width_starting_point_roi,
                                        height_starting_point_roi,
                                        scale_width,
                                        scale_height,
                                        image,
                                        GREEN)

            # Note: this is for when lane_dir and slot_dir are using the user-clicked point of reference
            if lane_dir not in ['u', 'd', 'l', 'r', 'c']:
                lane_dir_reference_point = lane_dir.split(',')
                lane_dir_reference_point = ([float(lane_dir_reference_point[0]), float(lane_dir_reference_point[1])])
                slot_dir_reference_point = slot_dir.split(',')
                slot_dir_reference_point = ([float(slot_dir_reference_point[0]), float(slot_dir_reference_point[1])])

    for i in range(len(list_of_points)):
        points = list_of_points[i]
        plot_selected_polygon(image, points, YELLOW_RGB)

    # Note: colorful polygons are the updates ones, with drift corrections possibly applied
    for i in range(len(list_of_points_updated)):
        points = list_of_points_updated[i]
        plot_selected_polygon(image, points, color_codes[i])

    for calib_point_index in range(len(video_config.calibration_point_coordinates)):
        calib_point = video_config.calibration_point_coordinates[calib_point_index]
        pos = calib_point.pixel
        plot_center_using_scale(pos,
                                width_starting_point_roi,
                                height_starting_point_roi,
                                scale_width,
                                scale_height,
                                image,
                                RED)
        w, h = pos
        w -= width_starting_point_roi
        h -= height_starting_point_roi
        w *= scale_width
        h *= scale_height
        w, h = int(w), int(h)

        cv2.putText(image, "calib_%s" % calib_point_index,
                    (w, h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    2)


def rescale(p, w, h, r_w, r_h):
    return int((p[0] - w) * r_w), int((p[1] - h) * r_h)


def plot_lane_id(frame, dict_lane_id, tracklets, scale_width, scale_height):
    num_dynamic_agents = sum([len(sublist) for sublist in tracklets])
    for sublist in tracklets:
        for tracklet in sublist:
            if tracklet.is_dynamic:
                num_dynamic_agents += 1

    if num_dynamic_agents <= 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_lane_id = 1
    font_scale_slot_id = 1
    line_thickness = 2

    for sublist in tracklets:
        for tracklet in sublist:
            if not tracklet.is_dynamic:
                continue
            traj = tracklet.traj
            pos = traj[-1]
            bbox = tracklet.bbox
            bound_box = bbox[-1]
            agent_id = tracklet.obj_id

            x, y = pos
            w, h = bound_box
            x = int(x * scale_width)
            w = int(w * scale_width)
            y = int(y * scale_height)
            h = int(h * scale_height)

            x1 = int(x - w * 0.5)
            y1 = int(y - h * 0.5)
            x2 = int(x + w * 0.5)
            y2 = int(y + h * 0.5)

            lane_id, fractional_lane_id, lane_width,\
            slot_id, fractional_slot_id, slot_length,\
            intersection_id, f_r, intersection_radius, f_phi, road_id = dict_lane_id[agent_id]

            if lane_id != 100:
                lane_id_label = "Lane %d/%.2f" % (lane_id, fractional_lane_id)

                cv2.putText(frame, lane_id_label,
                                (x2, y2),
                                font,
                                font_scale_lane_id,
                                RED,
                                line_thickness)

                slot_id_label = "Slot %s/%.2f" % (slot_id, fractional_slot_id)

                cv2.putText(frame, slot_id_label,
                            (x, y1),
                            font,
                            font_scale_slot_id,
                            RED,
                            line_thickness)

            elif intersection_id != -1:
                cv2.putText(frame,
                            "i_id %s r %.1f/%.1f phi %.1f" % (intersection_id, f_r, intersection_radius, f_phi),
                            (x2, y2),
                            font,
                            font_scale_lane_id,
                            RED,
                            line_thickness)
