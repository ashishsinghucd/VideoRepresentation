import os
import cv2
import csv
import argparse
import numpy as np
import mediapipe as mp
import logging
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    return results.pose_landmarks.landmark if results.pose_landmarks else None

def flatten_landmarks(landmarks):
    if not landmarks:
        return [np.nan] * (len(body_indices) * 4)
    return [coord for idx in body_indices for coord in (
        landmarks[idx].x, landmarks[idx].y, landmarks[idx].z, landmarks[idx].visibility)]

def process_video(video_path, output_path, summary_stats):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_path, f"{video_name}.csv")

    if os.path.exists(output_csv):
        logging.info(f"Skipping {video_name}, already processed.")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        data = []
        total_frames = 0
        detected_frames = 0

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames += 1
                landmarks = extract_pose_from_frame(frame, pose_model)
                if landmarks:
                    detected_frames += 1
                data.append(flatten_landmarks(landmarks))
        cap.release()

        os.makedirs(output_path, exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'{name}_{a}' for name in body_landmark_names for a in ['x', 'y', 'z', 'v']]
            writer.writerow(header)
            writer.writerows(data)

        detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
        summary_stats.append({
            "video_name": video_name,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": round(detection_rate, 4)
        })

        logging.info(f"{video_name} - Frames: {total_frames}, Detections: {detected_frames}, Rate: {detection_rate:.2%}")

    except Exception as e:
        logging.error(f"Error processing {video_name}: {e}")

def main(args):
    try:
        all_video_paths = []
        for root, _, files in os.walk(args.input_path):
            for f in files:
                if f.endswith(args.video_ext):
                    all_video_paths.append(os.path.join(root, f))

        logging.info(f"Found {len(all_video_paths)} video files.")
        summary_stats = []

        for path in tqdm(all_video_paths, desc="Processing videos"):
            process_video(path, args.output_path, summary_stats)

        # Save summary CSV
        summary_csv_path = os.path.join(args.output_path, "pose_detection_summary.csv")
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "total_frames", "detected_frames", "detection_rate"])
            writer.writeheader()
            writer.writerows(summary_stats)
        logging.info(f"Detection summary saved to: {summary_csv_path}")

    except Exception as e:
        logging.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose keypoints using MediaPipe and diagnose pose quality.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to root folder containing class subfolders with video files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output folder to save CSV pose files")
    parser.add_argument("--video_ext", type=str, default=".mp4",
                        help="Video file extension to look for")

    args = parser.parse_args()
    main(args)
