import os
import cv2
import csv
import argparse
import numpy as np
import mediapipe as mp
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

mp_pose = mp.solutions.pose

# Only body landmark indices (excluding face/toes, and without neck computation)
body_indices = [
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]

landmark_names = {
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow', 14: 'right_elbow',
    15: 'left_wrist', 16: 'right_wrist',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle',
}

def extract_pose_from_frame(image, pose_model):
    results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def flatten_landmarks(landmarks):
    if not landmarks:
        return [np.nan] * (len(body_indices) * 4)

    data = []
    for idx in body_indices:
        lm = landmarks[idx]
        data.extend([lm.x, lm.y, lm.z, lm.visibility])
    return data

def process_video(video_path, output_csv, pose_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_pose_from_frame(frame, pose_model)
        pose = flatten_landmarks(landmarks)
        data.append(pose)

    cap.release()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{landmark_names[idx]}_{dim}' for idx in body_indices for dim in ['x', 'y', 'z', 'v']]
        writer.writerow(header)
        writer.writerows(data)

def parse_split_file(split_file):
    with open(split_file, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def main(args):
    try:
        for split in ['train', 'val', 'test']:
            split_file = os.path.join(args.list_path, f'{split}.txt')
            video_label_list = parse_split_file(split_file)
            logging.info(f"Found {len(video_label_list)} videos in {split} split.")

            with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose_model:
                for video_id, label in tqdm(video_label_list, desc=f'Processing {split}'):
                    video_file = os.path.join(args.data_path, f"{video_id}.mp4")
                    output_csv = os.path.join(args.output_path, split, f"{video_id}.csv")

                    if os.path.exists(output_csv):
                        logging.info(f"Skipping {video_id}, already processed.")
                        continue

                    try:
                        process_video(video_file, output_csv, pose_model)
                        logging.info(f"Saved pose CSV: {output_csv}")
                    except Exception as e:
                        logging.error(f"Error processing {video_id}: {e}")

    except Exception as e:
        logging.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose keypoints (1 person) using MediaPipe.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to folder containing video files (*.mp4)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output folder to save CSV pose files")
    parser.add_argument("--list_path", type=str, required=True,
                        help="Folder where train.txt / val.txt / test.txt are stored")
    args = parser.parse_args()
    main(args)
