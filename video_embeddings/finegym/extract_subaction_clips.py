import os
import argparse
import logging
import time
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


def find_video_file(video_name, video_dir):
    """
    Finds the full path of a video file by checking for .mkv and .mp4 extensions.

    Args:
        video_name (str): The name of the video file without extension.
        video_dir (str): The directory where the videos are stored.

    Returns:
        str or None: The full path to the video file if found, otherwise None.
    """
    for ext in ['.mkv', '.mp4']:
        video_path = os.path.join(video_dir, f"{video_name}{ext}")
        if os.path.exists(video_path):
            return video_path
    return None


def extract_clip(video_path, output_path, start_time, end_time):
    """
    Extracts a subclip from a video file using a direct ffmpeg command for speed.

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
        return False


def process_split_file(file_path, video_dir, output_dir):
    """
    Processes a single split file (train.txt or val.txt) to extract all specified clips.

    Args:
        file_path (str): The full path to the split file.
        video_dir (str): The directory containing source videos.
        output_dir (str): The directory to save extracted clips.

    Returns:
        tuple: A tuple containing counts of (success, failure, video_not_found, skipped).
    """
    if not os.path.exists(file_path):
        logging.warning(f"Split file not found: {file_path}. Skipping.")
        return 0, 0, 0, 0

    success_count = 0
    failure_count = 0
    video_not_found_count = 0
    skipped_count = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

    logging.info(f"Processing {len(lines)} entries from {os.path.basename(file_path)}...")

    for line in lines:
        try:
            parts = line.strip().split()
            if not parts:
                continue

            full_clip_id = parts[0]

            # Construct the final output path first to check if it exists
            output_filename = f"{full_clip_id}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                # logging.info(f"Clip already exists: {output_path}. Skipping.")
                skipped_count += 1
                continue

            # Parse the identifier string
            # e.g., 0LtLS9wROrk_E_002407_002435_A_0003_0005
            id_parts = full_clip_id.split('_')
            source_video_name = "_".join(id_parts[:-3])  # e.g., 0LtLS9wROrk_E_002407_002435
            start_time_str = id_parts[-2]  # e.g., 0003
            end_time_str = id_parts[-1]  # e.g., 0005

            start_time = int(start_time_str)
            end_time = int(end_time_str)

            # Find the source video file (.mp4 or .mkv)
            source_video_path = find_video_file(source_video_name, video_dir)
            if not source_video_path:
                logging.warning(
                    f"Source video not found for clip ID: {full_clip_id}. Searched for '{source_video_name}.[mp4|mkv]'")
                video_not_found_count += 1
                continue

            # Extract the clip
            if extract_clip(source_video_path, output_path, start_time, end_time):
                success_count += 1
            else:
                failure_count += 1

        except (IndexError, ValueError) as e:
            logging.error(f"Could not parse line: '{line.strip()}'. Error: {e}")
            failure_count += 1
        except Exception as e:
            logging.error(f"An unexpected error occurred for line '{line.strip()}'. Error: {e}")
            failure_count += 1

    return success_count, failure_count, video_not_found_count, skipped_count


def main(args):
    """
    Main function to orchestrate the video clip extraction process.
    """
    start_process_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_clips_path, exist_ok=True)

    total_success = 0
    total_failure = 0
    total_not_found = 0
    total_skipped = 0

    # Process both train and validation split files
    for split_name in ['gym99_train_element_v1.1.txt', 'gym99_val_element.txt']:
        file_path = os.path.join(args.splits_folder, split_name)
        logging.info(f"--- Starting processing for {split_name} ---")

        s, f, nf, sk = process_split_file(file_path, args.input_videos_path, args.output_clips_path)

        total_success += s
        total_failure += f
        total_not_found += nf
        total_skipped += sk

    end_process_time = time.time()

    # --- Print Final Metrics ---
    logging.info("--- Extraction Process Finished ---")
    logging.info(f"Successfully Extracted New Clips: {total_success}")
    logging.info(f"Clips Already Existing (Skipped): {total_skipped}")
    logging.info(f"Failed Extractions: {total_failure}")
    logging.info(f"Clips Skipped (Source Video Not Found): {total_not_found}")
    logging.info(f"Total Processing Time: {end_process_time - start_process_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract FineGym sub-action clips based on train/val.txt files.")
    parser.add_argument('--splits_folder', type=str, required=True,
                        help="Path to the folder containing train.txt and val.txt.")
    parser.add_argument('--input_videos_path', type=str, required=True,
                        help="Path to the folder containing the source event-level videos (e.g., ID_E_XXXX_YYYY.mp4).")
    parser.add_argument('--output_clips_path', type=str, required=True,
                        help="Path to the folder where the final sub-action clips will be saved.")

    # Example usage:
    # python your_script_name.py --splits_folder ./annotations --input_videos_path ./event_clips --output_clips_path ./subaction_clips

    args = parser.parse_args()
    main(args)
