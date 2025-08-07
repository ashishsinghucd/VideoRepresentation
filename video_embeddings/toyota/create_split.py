import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define training person IDs
TRAIN_IDS = {'p03', 'p04', 'p06', 'p07', 'p09', 'p12', 'p13', 'p15', 'p17', 'p19', 'p25'}

def parse_filename(filename):
    """Parses the filename to extract class name and person ID."""
    try:
        name_parts = filename.stem.split('_')
        class_name = name_parts[0]
        person_id = name_parts[1]
        return class_name, person_id
    except Exception as e:
        logging.error(f"Error parsing filename {filename.name}: {e}")
        return None, None

def generate_split(dataset_path, output_path):
    train_lines = []
    test_lines = []

    try:
        all_files = list(Path(dataset_path).glob("*.mp4"))
        if not all_files:
            logging.warning("No .mp4 files found in the folder.")
            return

        for file in all_files:
            class_name, person_id = parse_filename(file)
            if not class_name or not person_id:
                continue  # Skip bad entries

            line = f"{file.stem}, {class_name}, {person_id}"
            if person_id in TRAIN_IDS:
                train_lines.append(line)
            else:
                test_lines.append(line)

        # Ensure output path exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save to files
        with open(output_path / 'train.txt', 'w') as f_train:
            f_train.write('\n'.join(train_lines))

        with open(output_path / 'test.txt', 'w') as f_test:
            f_test.write('\n'.join(test_lines))

        # Print metrics
        logging.info(f"Total files: {len(all_files)}")
        logging.info(f"Training samples: {len(train_lines)}")
        logging.info(f"Testing samples: {len(test_lines)}")
        logging.info(f"Files saved in: {output_path.resolve()}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test split generator for Toyota dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder containing .mp4 files")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="Folder to save train/test files")

    args = parser.parse_args()
    output_folder = args.output_path if args.output_path else args.dataset_path
    generate_split(args.dataset_path, output_folder)



"""
INFO: Total files: 16115
INFO: Training samples: 10682
INFO: Testing samples: 5433
"""