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
    try:
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # T x C x H x W
        if video.shape[0] < num_frames:
            return None
        indices = torch.linspace(0, video.shape[0] - 1, num_frames).long()
        sampled = video[indices]
        return sampled
    except Exception as e:
        logging.error(f"Error loading video {video_path}: {e}")
        return None

def load_video_paths(list_path, base_path):
    try:
        with open(list_path, "r") as f:
            lines = f.readlines()
        # Expecting format: path label
        video_paths = [os.path.join(base_path, "videos", line.strip().split()[0] + ".mp4") for line in lines]
        return video_paths
    except Exception as e:
        logging.error(f"Error reading file {list_path}: {str(e)}")
        return []

def extract_embeddings_batch(video_paths, model, processor, device, output_folder, batch_size=4):
    success = 0
    total = len(video_paths)

    for i in tqdm(range(0, total, batch_size), desc="Batch Processing"):
        batch_paths = video_paths[i:i+batch_size]
        clips = []
        valid_paths = []

        for path in batch_paths:
            clip = read_video_clip(path)
            if clip is not None:
                clips.append(list(clip))  # Convert to list of frames
                valid_paths.append(path)

        if not clips:
            continue

        try:
            # processed_clips = [clip.permute(0, 2, 3, 1).numpy().astype(np.uint8) for clip in clips]
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

def main(data_dir, output_path, split="train", batch_size=4):
    start_time = time.time()
    logging.info(f"Starting embedding extraction for split: {split}")

    list_file = os.path.join(data_dir, f"Splits/{split}.txt")
    video_list = load_video_paths(list_file, data_dir)
    total_videos = len(video_list)
    logging.info(f"Found {total_videos} videos for {split} split")
    logging.info(f"First file path {video_list[0]} videos for split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    output_folder = os.path.join(output_path, f"{split}_embeddings")
    os.makedirs(output_folder, exist_ok=True)

    success_count = extract_embeddings_batch(video_list, model, processor, device, output_folder, batch_size)

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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for embedding extraction")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path to save the embeddings")
    args = parser.parse_args()

    main(args.data_dir, args.output_path, args.split, args.batch_size)
