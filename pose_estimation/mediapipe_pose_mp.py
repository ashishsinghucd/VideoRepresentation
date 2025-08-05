import os
import cv2
import csv
import argparse
import numpy as np
import mediapipe as mp
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

mp_pose = mp.solutions.pose

# Selected body landmark indices (excluding face/toes, no neck)
body_indices = [
    11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
]

landmark_names_full = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

body_landmark_names = [landmark_names_full[i] for i in body_indices]

def extract_pose_from_frame(image, pose_model):
    results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    else:
        return None

def flatten_landmarks(landmarks):
    if not landmarks:
        return [np.nan] * (len(body_indices) * 4)
    return [coord for idx in body_indices for coord in (
        landmarks[idx].x, landmarks[idx].y, landmarks[idx].z, landmarks[idx].visibility)]

def process_single_video(video_id, label, args):
    video_file = os.path.join(args.data_path, f"{video_id}.mp4")
    output_csv = os.path.join(args.output_path, args.split, f"{video_id}.csv")

    if os.path.exists(output_csv):
        logging.info(f"Skipping {video_id}, already processed.")
        return

    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_file}")

        data = []
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = extract_pose_from_frame(frame, pose_model)
                data.append(flatten_landmarks(landmarks))
        cap.release()

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'{name}_{a}' for name in body_landmark_names for a in ['x', 'y', 'z', 'v']]
            writer.writerow(header)
            writer.writerows(data)

        logging.info(f"Saved pose CSV: {output_csv}")

    except Exception as e:
        logging.error(f"Error processing {video_id}: {e}")

def parse_split_file(split_file):
    with open(split_file, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def main(args):
    try:
        for split in ["train", "val", "test"]:
            args.split = split  # Add split info to args for use in process_single_video
            split_file = os.path.join(args.list_path, f'{split}.txt')
            video_label_list = parse_split_file(split_file)
            logging.info(f"Found {len(video_label_list)} videos in {split} split.")

            func = partial(process_single_video, args=args)
            with Pool(processes=cpu_count()) as pool:
                list(tqdm(pool.imap_unordered(lambda x: func(*x), video_label_list),
                          total=len(video_label_list),
                          desc=f'Processing {split}'))

    except Exception as e:
        logging.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose keypoints (1 person) using MediaPipe with multiprocessing.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to folder containing video files (*.mp4)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output folder to save CSV pose files")
    parser.add_argument("--list_path", type=str, required=True,
                        help="Folder where train.txt / val.txt / test.txt are stored")

    args = parser.parse_args()
    main(args)
