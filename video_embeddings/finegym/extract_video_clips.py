import os
import json
import configparser
import logging
import time
from moviepy.editor import VideoFileClip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting

# --- 1. Setup Logging ---
# Configure logging to show timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Also print to console
    ]
)


def find_video_file(video_id, video_dir):
    """
    Finds the full path of a video file by checking for .mkv and .mp4 extensions.

    Args:
        video_id (str): The unique identifier of the video (e.g., "0LtLS9wROrk").
        video_dir (str): The directory where the videos are stored.

    Returns:
        str or None: The full path to the video file if found, otherwise None.
    """
    for ext in ['.mkv', '.mp4']:
        video_path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path
    return None


def extract_clip(video_path, output_path, start_time, end_time):
    """
    Extracts a subclip from a video file using moviepy.
    Uses direct ffmpeg command for speed and to avoid re-encoding issues.

    Args:
        video_path (str): Path to the source video.
        output_path (str): Path to save the extracted clip.
        start_time (float): Start time of the clip in seconds.
        end_time (float): End time of the clip in seconds.
    """
    try:
        # Using moviepy's internal ffmpeg call for robust, fast, and lossless cutting
        ffmpeg_binary = get_setting("FFMPEG_BINARY")
        cmd = [
            ffmpeg_binary,
            "-y",  # Overwrite output file if it exists
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(end_time - start_time),
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "copy",  # Copy audio stream without re-encoding
            "-map", "0",  # Copy all streams
            output_path,
        ]

        subprocess_call(cmd, logger=None)  # Use None to prevent moviepy's verbose logging
        logging.info(f"Successfully extracted clip to {os.path.basename(output_path)}")
        return True
    except Exception as e:
        logging.error(f"Failed to extract clip {os.path.basename(output_path)}. Error: {e}")
        # Fallback to slower method if direct copy fails (e.g., format issues)
        logging.info(f"Attempting fallback extraction for {os.path.basename(output_path)}...")
        try:
            with VideoFileClip(video_path) as video:
                subclip = video.subclip(start_time, end_time)
                subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            logging.info(f"Successfully extracted clip using fallback method.")
            return True
        except Exception as fallback_e:
            logging.error(f"Fallback extraction also failed for {os.path.basename(output_path)}. Error: {fallback_e}")
            return False


def main():
    """
    Main function to read annotations and extract video clips.
    """
    config = configparser.ConfigParser()
    config_file = 'config.ini'

    if not os.path.exists(config_file):
        logging.error(f"Configuration file '{config_file}' not found. Please create it.")
        return

    config.read(config_file)

    # --- Read paths from config file ---
    try:
        video_dir = config.get('PATHS', 'input_path')
        splits_dir = config.get('PATHS', 'splits_path')
        output_dir = config.get('PATHS', 'output_path')
    except configparser.NoOptionError as e:
        logging.error(f"Configuration error: {e}. Please check your config.ini file.")
        return

    # The main annotation file for FineGym
    annotation_file = os.path.join(splits_dir, 'finegym_annotation_info_v1.1.json')

    if not os.path.exists(annotation_file):
        logging.error(f"Annotation file not found at: {annotation_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Load annotations ---
    logging.info(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    logging.info("Annotations loaded.")

    # --- Initialize metrics ---
    total_events = 0
    success_count = 0
    failure_count = 0
    video_not_found_count = 0
    start_process_time = time.time()

    # --- Process each video and its events ---
    for video_id, video_data in annotations.items():
        source_video_path = find_video_file(video_id, video_dir)

        if not source_video_path:
            num_events_in_missing_video = len(video_data)
            logging.warning(
                f"Source video not found for ID: {video_id}. Skipping {num_events_in_missing_video} events.")
            video_not_found_count += num_events_in_missing_video
            total_events += num_events_in_missing_video
            continue

        logging.info(f"Processing video: {source_video_path}")

        for event_id, event_data in video_data.items():
            total_events += 1
            try:
                # The 'timestamps' key contains a list with one sublist [start, end]
                timestamps = event_data['timestamps'][0]
                start_time, end_time = float(timestamps[0]), float(timestamps[1])

                # Construct the unique identifier for the clip
                # e.g., 0LtLS9wROrk_E_002407_002435
                clip_name = f"{video_id}_{event_id}"
                output_path = os.path.join(output_dir, f"{clip_name}.mp4")

                if os.path.exists(output_path):
                    logging.info(f"Clip already exists: {output_path}. Skipping.")
                    success_count += 1
                    continue

                if extract_clip(source_video_path, output_path, start_time, end_time):
                    success_count += 1
                else:
                    failure_count += 1

            except (KeyError, IndexError, TypeError) as e:
                logging.error(f"Error parsing annotation for event {event_id} in video {video_id}. Error: {e}")
                failure_count += 1
            except Exception as e:
                logging.error(f"An unexpected error occurred for event {event_id}. Error: {e}")
                failure_count += 1

    end_process_time = time.time()

    # --- Print Final Metrics ---
    logging.info("--- Extraction Process Finished ---")
    logging.info(f"Total Events in Annotations: {total_events}")
    logging.info(f"Successfully Extracted Clips: {success_count}")
    logging.info(f"Failed Extractions: {failure_count}")
    logging.info(f"Events Skipped (Video Not Found): {video_not_found_count}")
    logging.info(f"Total Processing Time: {end_process_time - start_process_time:.2f} seconds")


if __name__ == '__main__':
    main()


"""
/home/x_ashsi            9.4 GiB  20.0 GiB    30.0 GiB       56155  1000000     1500000
/proj/tinyml_htg_ltu     3.3 TiB   4.9 TiB     6.1 TiB     5553676  6000000     9000000

2025-08-07 14:34:23,738 - INFO - --- Extraction Process Finished ---
2025-08-07 14:34:23,738 - INFO - Total Events in Annotations: 12818
2025-08-07 14:34:23,738 - INFO - Successfully Extracted Clips: 12469
2025-08-07 14:34:23,738 - INFO - Failed Extractions: 0
2025-08-07 14:34:23,738 - INFO - Events Skipped (Video Not Found): 349
2025-08-07 14:34:23,738 - INFO - Total Processing Time: 3084.50 seconds

"""