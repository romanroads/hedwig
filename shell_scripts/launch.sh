#!/bin/bash

export MODEL_PATH="../data/"
export DATA_PATH='../data/'
export DATA_FILE="test.mp4"

set -e

cd ./python/ && \
python main.py --video_path $DATA_PATH --video_file $DATA_FILE \
--min_frame 0 --max_frame -1  --logging info \
--center_dot_size 5 --traj_width 2 --agent_id_text_size 2 --agent_id_font_size 0.9 --agent_id_offset_w 40 \
-x 1500 -y 150 --tracking_x 3000 --tracking_y 300 -a 3000 -b 300 \
--tracklet_traj --commit_min_viable_data --stablization_correction --model_path $MODEL_PATH --detection_frame_gap 2 \
--reg_unsub_zone --stablization_zone \
--thresh_det "Sedan:0.6,Truck:0.6,Motorcycle:0.7,Bicycle:0.7,Pedestrian:0.8,RoadFeatures:0.8"
