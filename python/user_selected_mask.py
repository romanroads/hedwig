import numpy as np
import cv2

from constant_values import WHITE


def get_masked_image(frame, list_frac_pixel_cords_roi):
    height, width, depth = frame.shape

    arr_pixel_cords = np.array([(int(p[0] * width), int(p[1] * height)) for p in list_frac_pixel_cords_roi],
                               dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillConvexPoly(mask, arr_pixel_cords, WHITE)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    return res


def get_masked_image_with_roi(frame, list_frac_pixel_cords_roi, rect_roi):
    height, width, depth = frame.shape
    x, y, w, h = rect_roi

    arr_pixel_cords = np.array([(int(p[0] * width), int(p[1] * height)) for p in list_frac_pixel_cords_roi],
                               dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)

    # cv2.fillConvexPoly(mask, arr_pixel_cords, WHITE)
    cv2.fillPoly(mask, [arr_pixel_cords], WHITE)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    return res[y:y+h, x:x+w]


def get_masked_image_with_roi_return_mask(frame, list_frac_pixel_cords_roi, rect_roi):
    height, width, depth = frame.shape
    x, y, w, h = rect_roi

    arr_pixel_cords = np.array([(int(p[0] * width), int(p[1] * height)) for p in list_frac_pixel_cords_roi],
                               dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)

    # cv2.fillConvexPoly(mask, arr_pixel_cords, WHITE)
    cv2.fillPoly(mask, [arr_pixel_cords], WHITE)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    return res[y:y+h, x:x+w], mask


def is_pixel_in_mask_generated_by_roi(mask, point_x, point_y):
    if point_y > mask.shape[0]:
        return False
    if point_x > mask.shape[1]:
        return False
    return mask[int(point_y)][int(point_x)] > 0


def get_rect_roi(frame, list_frac_pixel_cords_roi):
    height, width, depth = frame.shape

    if len(list_frac_pixel_cords_roi) <= 0:
        return 0, 0, width, height

    arr_pixel_cords = np.array([(int(p[0] * width), int(p[1] * height)) for p in list_frac_pixel_cords_roi],
                               dtype=np.int32)
    rect = cv2.boundingRect(arr_pixel_cords)

    return rect
