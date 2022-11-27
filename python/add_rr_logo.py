import os
from pathlib import Path
import cv2

from constant_values import *

logo_read = False
img = None
img_customer = None
mask = None
mask_c = None
mask_inv = None
mask_inv_c = None
fg_from_log = None
fg_from_log_c = None
margin_col = 0
margin_row = 0


def add_rr_logo(frame, top_right=True, logo_pos=""):
    """
    here the frame is the dedicated frame to write out to mp4 files
    :param frame:
    :param top_right:
    :param logo_pos:
    :return:
    """
    global logo_read, img, img_customer, mask, mask_c, mask_inv, mask_inv_c, fg_from_log, fg_from_log_c, margin_col,\
        margin_row

    # Note: these are dimensions for the output frame
    row_frame, col_frame, c_frame = frame.shape

    scale_percent = 15.  # percentage of output frame for the size of the logo image part
    scale_percent_margin = 100.  # percentage of logo image frame for the margin for the logo image part

    if not logo_read:
        dirname, filename = os.path.split(os.path.abspath(__file__))
        path = Path(dirname)
        path_to_logo_folder = os.path.join(path.parent, ARTIFACT_FOLDER_NAME)

        path_to_logo = os.path.join(path_to_logo_folder, LOGO_FILE_NAME)
        # path_to_logo_customer = os.path.join(path_to_logo_folder, LOGO_FILE_CUSTOMER_2_NAME)

        img = cv2.imread(path_to_logo)

        # Note: this is aspect ratio for the logo image
        aspect_ratio = img.shape[1] / img.shape[0]

        # img_customer = cv2.imread(path_to_logo_customer)
        # img_customer = img_customer[100:-100, :]

        # Note: we would like keep the original logo image aspect ratio, so we use only the width of the re-scale
        width_logo = int(col_frame * scale_percent / 100)
        height_logo = int(width_logo / aspect_ratio)

        height_logo_alter = int(row_frame * scale_percent / 100)
        width_logo_alter = int(height_logo_alter * aspect_ratio)

        # Note: we select the one that is smaller
        if width_logo_alter < width_logo:
            width_logo = width_logo_alter
            height_logo = height_logo_alter

        dim = (width_logo, height_logo)

        min_log_width_height = min(width_logo, height_logo)
        margin_col = int(min_log_width_height * (scale_percent_margin / 100.))
        margin_row = int(min_log_width_height * (scale_percent_margin / 100.))

        img_ori = cv2.resize(img, dim)
        # img_customer_ori = cv2.resize(img_customer, dim)

        # note for some reason our white logo does not work, have to reverse the black logo to make the white logo...
        img = cv2.bitwise_not(img_ori)
        rr_logo_white = img.copy()

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        fg_from_log = cv2.bitwise_and(rr_logo_white, rr_logo_white, mask=mask)

        # fg_from_log_c = img_customer_ori

        logo_read = True

    # Note: these are the dimensions of logo part of image after resizing them to match to the size of output frame
    rows, cols, channels = img.shape

    if len(logo_pos) > 0:
        if logo_pos == "bottom_left":
            start_row = row_frame - margin_row - rows
            start_col = margin_col
            # start_row_customer = row_frame - margin_row - rows
            # start_col_customer = col_frame - margin_col - cols
        elif logo_pos == "top_left":
            start_row = margin_row
            start_col = margin_col
        elif logo_pos == "bottom_right":
            start_row = row_frame - margin_row - rows
            start_col = col_frame - margin_col - cols
        elif logo_pos == "top_right":
            start_row = margin_row
            start_col = col_frame - margin_col - cols

    else:
        if top_right:
            margin_row = 100
        else:
            margin_row = row_frame - 300

    roi = frame[start_row:start_row + rows, start_col:start_col + cols]

    bg_from_image = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(bg_from_image, fg_from_log)

    # Note: frame is the drone or cctv image, while dst is the combined logo and drone or cctv image within a ROI
    frame[start_row:start_row + rows, start_col:start_col + cols] = dst


