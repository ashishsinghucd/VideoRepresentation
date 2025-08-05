import os
import csv
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIGURE LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("kinetics_downloader.log"),
        logging.StreamHandler()
    ]
)

# === YOUR EXERCISE CLASSES ===
target_classes = [
    'clean and jerk', 'deadlifting', 'snatch weight lifting', 'bench pressing',
    'front raises', 'pull ups', 'push up', 'mountain climber (exercise)',
    'situp', 'squat', 'jumping jacks', 'battle rope training', 'lunge',
    'exercising with an exercise ball', 'rope pushdown', 'stretching arm',
    'stretching leg', 'running on treadmill', 'jogging', 'skipping rope',
    'climbing a rope', 'climbing ladder', 'climbing tree', 'parkour'
]



def load_kinetics_annotations(csv_file, target_classes):
    """
    Load and filter Kinetics annotations.
    """
    filtered = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip().lower()
            if label in [cls.lower() for cls in target_classes]:
                filtered.append({
                    'youtube_id': row['youtube_id'],
                    'start_time': row['time_start'],
                    'end_time': row['time_end'],
                    'label': label
                })
    return filtered


def download_clip(ann, output_dir, retries=3):
    """
    Download and trim clip. Retry on failure.
    """
    youtube_id = ann['youtube_id']
    start_time = ann['start_time']
    end_time = ann['end_time']
    label = ann['label']

    filename = f"{label.replace(' ', '_')}_{youtube_id}_{start_time}_{end_time}.mp4"
    output_path = os.path.join(output_dir, filename)
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    if os.path.exists(output_path):
        logging.info(f"Already exists: {filename}")
        return True

    command = [
        'yt-dlp',
        url,
        '--quiet',
        '--no-warnings',
        '--external-downloader', 'ffmpeg',
        '--external-downloader-args', f"-ss {start_time} -to {end_time} -c copy",
        '-o', output_path
    ]

    attempt = 0
    while attempt < retries:
        try:
            subprocess.run(command, check=True)
            logging.info(f"Downloaded: {filename}")
            return True
        except subprocess.CalledProcessError:
            attempt += 1
            logging.warning(f"Retry {attempt}/{retries} failed for {url}")

    logging.error(f"Failed after {retries} attempts: {url}")
    return False


def download_kinetics_subset(csv_file, target_classes, output_dir, num_workers=4):
    """
    Download clips using parallel workers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = load_kinetics_annotations(csv_file, target_classes)
    total = len(annotations)
    logging.info(f"Found {total} clips for target classes.")

    downloaded_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_clip, ann, output_dir): ann for ann in annotations}

        for i, future in enumerate(as_completed(futures), start=1):
            ann = futures[future]
            try:
                success = future.result()
                if success:
                    downloaded_count += 1
            except Exception as e:
                logging.error(f"Unexpected error: {e} for {ann['youtube_id']}")

            logging.info(f"Progress: {downloaded_count}/{total} clips downloaded")


if __name__ == "__main__":
    # === USER CONFIG ===
    csv_file = "./Kinetics600/kinetics-600_train.csv"  # path to your CSV file
    output_dir = "./Kinetics600/kinetics_exercise_subset"
    num_workers = 4  # adjust based on your CPU/network

    download_kinetics_subset(csv_file, target_classes, output_dir, num_workers)
