import os
import time
import logging
from pathlib import Path
from transformers import AutoProcessor, VideoMAEModel
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("FitnessAQA/videomae_embedding.log"),
        logging.StreamHandler()
    ]
)

def read_video_clip(video_path, num_frames=16):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames < num_frames:
            return None
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        clip = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
        return clip.astype(np.uint8)
    except Exception as e:
        logging.error(f"Error loading video {video_path}: {e}")
        return None

def load_video_paths(list_path, base_path):
    try:
        with open(list_path, "r") as f:
            lines = f.readlines()
        video_paths = [os.path.join(base_path, "videos", line.strip().split()[0] + ".mp4") for line in lines]
        return video_paths[:4]
    except Exception as e:
        logging.error(f"Error reading file {list_path}: {str(e)}")
        return []

def extract_embeddings(video_paths, model, processor, device, output_folder):
    success = 0

    for path in tqdm(video_paths, desc="Processing Videos"):
        clip = read_video_clip(path)
        if clip is None:
            continue

        try:
            inputs = processor([clip], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

            name = Path(path).stem + ".pt"
            torch.save(embedding, os.path.join(output_folder, name))
            success += 1

        except Exception as e:
            logging.error(f"Processing error for {path}: {e}")

    return success

def main(data_dir, split="train"):
    start_time = time.time()
    logging.info(f"Starting embedding extraction for split: {split}")

    list_file = os.path.join(data_dir, f"Splits/{split}.txt")
    video_list = load_video_paths(list_file, data_dir)
    total_videos = len(video_list)
    if total_videos == 0:
        logging.error("No videos found. Exiting.")
        return

    logging.info(f"Found {total_videos} videos for {split} split")
    logging.info(f"First file path: {video_list[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    output_folder = os.path.join(data_dir, f"{split}_embeddings")
    os.makedirs(output_folder, exist_ok=True)

    success_count = extract_embeddings(video_list, model, processor, device, output_folder)

    total_time = time.time() - start_time
    logging.info(f"Done! Successfully processed {success_count}/{total_videos} videos.")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to folder containing train.txt / val.txt / test.txt and videos")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process: train / val / test")
    args = parser.parse_args()

    main(args.data_dir, args.split)
