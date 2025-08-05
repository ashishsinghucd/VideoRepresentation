import os
import time
import logging
from pathlib import Path
from transformers import AutoProcessor, VideoMAEModel
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("videomae_embedding.log"),
        logging.StreamHandler()
    ]
)

def load_frame_clip(frame_dir, video_id, num_frames=None):
    try:
        pattern = re.compile(rf"{re.escape(video_id)}_(\d+)\.jpg$", re.IGNORECASE)
        frame_files = []
        for fname in os.listdir(frame_dir):
            match = pattern.match(fname)
            if match:
                frame_files.append((int(match.group(1)), os.path.join(frame_dir, fname)))

        frame_files.sort()  # Sort by frame number

        if len(frame_files) == 0:
            return None

        if num_frames is None or num_frames >= len(frame_files):
            selected_files = [f[1] for f in frame_files]
        else:
            indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
            selected_files = [frame_files[i][1] for i in indices]

        clip = [np.array(Image.open(f).convert("RGB")) for f in selected_files]
        return clip
    except Exception as e:
        logging.error(f"Error loading frames for video {video_id}: {e}")
        return None

def load_video_ids(list_path):
    try:
        with open(list_path, "r") as f:
            lines = f.readlines()
        video_ids = [line.strip().split()[0] for line in lines]
        return video_ids
    except Exception as e:
        logging.error(f"Error reading file {list_path}: {str(e)}")
        return []

def pad_clip(clip, target_length):
    """Pads the clip by repeating the last frame."""
    if len(clip) >= target_length:
        return clip
    last_frame = clip[-1]
    return clip + [last_frame] * (target_length - len(clip))

def extract_embeddings_batch(video_ids, model, processor, device, output_folder, frame_dir, batch_size=4, num_frames=None):
    success = 0
    total = len(video_ids)

    for i in tqdm(range(0, total, batch_size), desc="Batch Processing"):
        batch_ids = video_ids[i:i+batch_size]
        clips = []
        valid_ids = []

        max_len = 0
        loaded_clips = {}

        for vid in batch_ids:
            clip = load_frame_clip(frame_dir, vid, num_frames=num_frames)
            if clip is not None:
                loaded_clips[vid] = clip
                valid_ids.append(vid)
                if len(clip) > max_len:
                    max_len = len(clip)

        if not loaded_clips:
            continue

        for vid in valid_ids:
            clip = loaded_clips[vid]
            padded_clip = pad_clip(clip, max_len)
            resized_clip = [np.array(Image.fromarray(frame).resize((224, 224))) for frame in padded_clip]
            clips.append(resized_clip)

        try:
            inputs = processor(clips, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

            for emb, vid in zip(embeddings, valid_ids):
                torch.save(emb, os.path.join(output_folder, vid + ".pt"))
                success += 1

        except Exception as e:
            logging.error(f"Batch processing error: {e}")

    return success



def main(data_dir, output_path, split="train", batch_size=4, num_frames=None):
    start_time = time.time()
    logging.info(f"Starting embedding extraction for split: {split}")

    list_file = os.path.join(data_dir, f"Splits/Splits_Lumbar_Error/{split}.txt")
    video_ids = load_video_ids(list_file)
    total_videos = len(video_ids)
    if total_videos == 0:
        logging.error("No video IDs found. Exiting.")
        return

    logging.info(f"Found {total_videos} video IDs for {split} split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    frame_dir = os.path.join(data_dir, "barbellrow_images_raw")
    output_folder = os.path.join(output_path, f"{split}_embeddings")
    os.makedirs(output_folder, exist_ok=True)

    success_count = extract_embeddings_batch(video_ids, model, processor, device, output_folder, frame_dir, batch_size, num_frames)

    total_time = time.time() - start_time
    logging.info(f"Done! Successfully processed {success_count}/{total_videos} videos.")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to folder containing train.txt / val.txt / test.txt and frames")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process: train / val / test")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for embedding extraction")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of frames to sample from each video (default: use all available frames)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path to save the embeddings")
    args = parser.parse_args()


    main(args.data_dir, args.output_path, args.split, args.batch_size, args.num_frames)

