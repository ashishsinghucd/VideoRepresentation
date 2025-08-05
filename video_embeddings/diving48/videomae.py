import os
import time
import logging
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch
import torchvision.io as io
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("videomae_embedding.log"),
        logging.StreamHandler()
    ]
)


def read_video_clip(video_path, num_frames=16):
    """
    Reads a video file and samples a fixed number of frames.
    Returns a tensor of shape (num_frames, C, H, W) or None if error.
    """
    try:
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # T x C x H x W
        num_available = video.shape[0]
        if num_available == 0:
            return None

        if num_available < num_frames:
            indices = torch.linspace(0, num_available - 1, num_available).long()
            sampled = video[indices]
            last_frame = sampled[-1].unsqueeze(0)
            padding = last_frame.repeat(num_frames - num_available, 1, 1, 1)
            sampled = torch.cat([sampled, padding], dim=0)
        else:
            indices = torch.linspace(0, num_available - 1, num_frames).long()
            sampled = video[indices]
        return sampled
    except Exception as e:
        logging.error(f"Error loading video {video_path}: {e}")
        return None

def get_all_video_paths(video_dir):
    """
    Recursively collects all .mp4 video file paths from a directory.
    """
    all_paths = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                all_paths.append(os.path.join(root, file))
    return all_paths

def extract_embeddings_batch(video_paths, model, processor, device, output_folder, batch_size=4):
    """
    Extracts embeddings for a batch of video paths and saves them to disk.
    """
    success = 0
    total = len(video_paths)

    for i in tqdm(range(0, total, batch_size), desc="Batch Processing"):
        batch_paths = video_paths[i:i+batch_size]
        clips = []
        valid_paths = []

        for path in batch_paths:
            clip = read_video_clip(path)
            if clip is not None:
                clips.append(list(clip))
                valid_paths.append(path)

        if not clips:
            continue

        try:
            inputs = processor(clips, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

            for emb, path in zip(embeddings, valid_paths):
                name = Path(path).stem + ".pt"
                torch.save(emb, os.path.join(output_folder, name))
                success += 1

        except Exception as e:
            logging.error(f"Batch processing error: {e}")

    return success

def main(video_dir, output_dir, batch_size=4):
    """
    Main function to extract embeddings from videos in a folder structure.

    Args:
        video_dir (str): Root path containing video files in class subfolders.
        output_dir (str): Directory to save .pt embedding files.
        batch_size (int): Number of videos to process per batch.
    """
    start_time = time.time()
    logging.info(f"Starting embedding extraction from folder: {video_dir}")

    video_list = get_all_video_paths(video_dir)
    total_videos = len(video_list)
    logging.info(f"Found {total_videos} video files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    success_count = extract_embeddings_batch(video_list, model, processor, device, output_dir, batch_size)

    total_time = time.time() - start_time
    logging.info(f"Done! Successfully processed {success_count}/{total_videos} videos.")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Root folder containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path to save the embeddings")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for embedding extraction")
    args = parser.parse_args()

    main(args.video_dir, args.output_dir, args.batch_size)
