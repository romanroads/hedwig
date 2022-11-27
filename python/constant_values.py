import cv2 as cv
import numpy as np


WINDOW_NAME = "Drone Perception & Tracking"

WARP_ALPHA = 0
WARP_BETA = -80
WARP_GAMMA = -40

VIDEO_NAME = "video_name"
RAW_IMAGE_NAME = "image"
DETECTION_IMAGE_NAME = "image_detection"
TRACKING_IMAGE_NAME = "image_tracking"
TRACKING_IMAGE_RGB_NAME = "image_tracking_rgb"
OUTPUT_IMAGE_NAME = "image_output"
FRAME_NUMBER_NAME = "frame_num"
FRAME_COUNT_NAME = "frame_count"
FPS_NAME = "fps"
ORI_FRAME_COUNT_NAME = "ori_frame_count"
ORI_FRAME_DIMEN_NAME = "ori_frame_dimen"
IMAGE_ID_NAME = "image_id"
PREDICTIONS_NAME = "predictions"
PROCESSED_PREDICTIONS_NAME = "processed_predictions"
PROCESSED_PREDICTIONS_CENTER_NAME = "processed_predictions_center"
PROCESSED_PREDICTIONS_LENGTH_NAME = "processed_predictions_length"
PROCESSED_PREDICTIONS_WIDTH_NAME = "processed_predictions_width"
PROCESSED_PREDICTIONS_POLY_POINTS_NAME = "processed_predictions_poly_points"
PROCESSED_PREDICTIONS_POLYGON_NAME = "processed_predictions_polygon"
PROCESSED_PREDICTIONS_MASK_DET_NAME = "processed_predictions_mask_det"
PROCESSED_PREDICTIONS_MASK_TRACK_NAME = "processed_predictions_mask_track"
PROCESSED_PREDICTIONS_MASK_COMBO_NAME = "processed_predictions_mask_combo"
PROCESSED_PREDICTIONS_POS_INDEX_NAME = "processed_predictions_pos_index"
PROCESSED_PREDICTIONS_DYNAMIC_NAME = "processed_predictions_is_dynamic"
PROCESSED_PREDICTIONS_BOX_WIDTH_NAME = "processed_predictions_box_width"
PROCESSED_PREDICTIONS_BOX_HEIGHT_NAME = "processed_predictions_box_height"
PROCESSED_PREDICTIONS_EIGENVECTOR_NAME = "processed_predictions_eigenvector"
PROCESSED_PREDICTIONS_FEATURE_POINTS_NAME = "processed_predictions_feature_points"
PROCESSED_PREDICTIONS_CLASS_NAME = "processed_predictions_class"
PROCESSED_PREDICTIONS_FEATURE_POINTS_COMBO_NAME = "processed_predictions_feature_points_combo"
PROCESSED_PREDICTIONS_SCORES_COMBO_NAME = "processed_predictions_scores_combo"
PROCESSED_PREDICTIONS_CONFIDENCE_NAME = "processed_predictions_confidence"

TRACKED_AGENTS_NAME = "tracked_agents"
MATCHED_AGENTS_NAME = "matched_agents"

TRACKING_STATISTICS = "tracking_statistics"

TRACKING_LOST_TYPE_UNSUB_ZONE = "in un-subscription zone"
TRACKING_LOST_TYPE_STALED_DET = "staled detection"

CONFIG_NAME = "configuration"
CALIBRATION_ZONE_POINTS_NAME = "calibration_zone_points"
CALIBRATION_ZONE_NAME = "calibration_zone"
CALIBRATION_IMAGE_MAPPING_NAME = "calibration_image_mapping"
CALIBRATION_POINT_A = "calibration_point_a"
CALIBRATION_POINT_B = "calibration_point_b"
CALIBRATION_RESOLUTION = "calibration_resolution"
STABLIZATION_ZONE_NAME = "stalization_zone"
STABLIZATION_SYSTEM_NAME = "stablization_system"

DIMENSIONS_NAME = "dimensions"

RELATIVE_MEAS_NAME = "relative_measurements"

LOCAL_TIMESTAMP = "local_timestamp"
GLOBAL_TIMESTAMP_START_CONFIG = "global_timestamp_start_config"
LOCATION_NAME_CONFIG = "location_name_config"

TRAINING_DATASET_NAME = "RRTrainingDataset"
TESTING_DATASET_NAME = "RRTestingDataset"
VALIDATION_DATASET_NAME = "RRValidationDataset"

RR_NET_MODEL_NAME = "model_rr_net.pth"

SAVED_VIDEO_FILE_PATH = "saved_video_file_path"
SAVED_VIDEO_FILE_CLOUD_KEY = "saved_video_file_cloud_key"

AWS_KEY_ID_ENV_NAME = "AWS_KEY_ID_ENV"
AWS_KEY_ENV_NAME = "AWS_KEY_ENV"
AWS_REGION_ENV_NAME = "AWS_REGION_ENV"
AWS_KEY_ID_ENV = ""
AWS_KEY_ENV = ""
AWS_REGION_ENV = ""
AWS_USER_UPLOAD_DATA_BUCKET_NAME = "user-upload-data"

DATA_FOLDER_ENV_NAME = "DRONEHOME"

AWS_BUCKET_NAME_TRAIN_DATA = "rr-training-data"
AWS_BUCKET_NAME_MODEL = "sample-model"


LK_PARAMS = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 40, 0.01))

FEATURE_PARAMS = dict(maxCorners=1000,
                      qualityLevel=0.02,
                      minDistance=4,
                      blockSize=4)

COLOR_MATRIX = np.random.randint(0, 255, (100, 3))

BLACK = (0, 0, 0)

WHITE = (255, 255, 255)

BLUE = (255, 0, 0)
BLUE_RGB = (0, 0, 255)

YELLOW = (255, 255, 0)
YELLOW_RGB = (0, 255, 255)

RED = (0, 0, 255)
RED_RGB = (255, 0, 0)

GREEN = (0, 255, 0)

X_BOUNDARY_THRESHOLD_RIGHT = 0.01  # 0.04
X_BOUNDARY_THRESHOLD_LEFT = 0.01  # 0.04

Y_BOUNDARY_THRESHOLD_TOP = 0.01  # 0.1 -> 0.01
Y_BOUNDARY_THRESHOLD_BOTTOM = 0.001

DETECTION_FRAME_GAP = 4

WIDTH_WINDOW = 1280

HEIGHT_WINDOW = 800

WINDOW_START_WIDTH = 10

FPS_OUTPUT = 30

INITIAL_TIME_WINDOW_TO_REGISTER_AGENTS = 0.5

TIME_THRESHOLD_TO_CLEAR_MEMORY_OF_REMOVED = 1

DIST_BETWEEN_TWO_POLYGON_TO_AVOID_DUP = 1.0

IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP = 0.1

ARTIFACT_FOLDER_NAME = "artifacts"

CONFIG_FILE_FOLDER_NAME = "config_files"

PROCESSED_VIDEO_FOLDER_NAME = "data"

LOGO_FILE_NAME = "ROMAN_ROADS_LOGO_BLACK.png"

LOGO_FILE_CUSTOMER_1_NAME = "faw.png"
LOGO_FILE_CUSTOMER_2_NAME = "li_auto.png"

WIDTH_FOR_LARGE_VEHICLE = 100

HEIGHT_FOR_LARGE_VEHICLE = 100

MAX_TRACKING_DIST_JUMP_SCALE = 3.0

MAX_TIMES_JUMP_ALLOWED = 3

MOVING_AVERAGE_WINDOW_SIZE = 5
SIZE_MOVING_WINDOW_AVERAGE_FOR_SPEED_SECONDS = 0.5

MOVING_AVERAGE_WINDOW_START = -10

# Note: this maps a broad range of semantic class names to the ones that our DNN cares about
CLASS_ID_UNDEFINED = 10

MAP_CATEGORY_NAME_SEMANTIC_CLASS = {
        'Car': 0,
        'Ego': 0,
        'Police': 0,
        'Suv': 1,
        'Truck': 2,
        'Trailer': 2,
        'Van': 3,
        'Bus': 4,
        'Motorcycle': 5,
        'Bicycle': 6,
        'Pedestrian': 7,
        'Scooter': 8,
        'AV': 9,
        'RoadGround': CLASS_ID_UNDEFINED,
        'RoadDivider': CLASS_ID_UNDEFINED,
        'RoadBound': CLASS_ID_UNDEFINED,
        'LaneMarker': CLASS_ID_UNDEFINED,
        'Building': CLASS_ID_UNDEFINED,
        'Gyro': CLASS_ID_UNDEFINED,
        'TrafficLight': CLASS_ID_UNDEFINED,
        'WheelChair': CLASS_ID_UNDEFINED,
        'Ambulance': CLASS_ID_UNDEFINED,
        'BodyPartHead': CLASS_ID_UNDEFINED,
        'BodyPartArm': CLASS_ID_UNDEFINED,
        'BodyPartChest': CLASS_ID_UNDEFINED,
        'BodyPartLeg': CLASS_ID_UNDEFINED,
        'Unknown': CLASS_ID_UNDEFINED,
        'RRRobot': CLASS_ID_UNDEFINED,
        'Tree': CLASS_ID_UNDEFINED
}

# Note: MAP_CATEGORY_ID_TO_NAME maps to what is really used in DNN while MAP_CATEGORY_NAME_SEMANTIC_CLASS is more
# flexible in terms of labeling parts on an object such as arm of a person...
# we need to do start from class id 0 due to the DNN input requirements
MAP_CATEGORY_ID_TO_NAME = {
        0: 'Sedan',
        1: 'SUV',
        2: 'Truck',
        3: 'Van',
        4: 'Bus',
        5: 'Motorcycle',
        6: 'Bicycle',
        7: 'Pedestrian',
        8: 'Scooter',
        9: 'AV',
        CLASS_ID_UNDEFINED: 'Undefined',
}

MAP_CATEGORY_ID_TO_COLOR = {
        0: (255, 0, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
        4: (255, 0, 0),
        5: (255, 0, 0),
        6: (255, 0, 0),
        7: (255, 0, 0),
        8: (255, 0, 0),
        9: (255, 0, 0),
        10: (255, 0, 0)
}

CATEGORY_NAMES_FOR_DYNAMIC_OBJECTS = ["Sedan", "Truck", "Motorcycle", "Bicycle", "Pedestrian"]

THRESHOLD_DETECTION_PRECISION = {
        "Sedan": 0.6,
        "Truck": 0.6,
        "Motorcycle": 0.7,
        "Bicycle": 0.7,
        "Pedestrian": 0.8,
        "RoadFeatures": 0.8,
}

FEATURE_POINT_MAPS = {
        "Sedan": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                },

        "Truck": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                },

        "Motorcycle": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                       },

        "Bicycle": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                  },

        "Pedestrian": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                       },

        "RoadFeatures": {"names": [],
                "flip_map": [],
                "connection_rules": [],
                "color_codes": [],
                "precisions": []
                       },
}

TIME_THRESHOLD_TO_DEFINE_TRACKLET_LOST = {
        "Sedan": 1.2,
        "Truck": 1.2,
        "Motorcycle": 2,
        "Bicycle": 4,
        "Pedestrian": 60000,
        "RoadFeatures": 60000,
}

ESCAPE_KEY = 27
