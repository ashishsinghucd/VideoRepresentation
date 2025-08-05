import os
import time
import torch
import numpy as np
import argparse
import logging

def convert_pt_to_npz(input_dir):
    start_time = time.time()
    count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pt'):
                pt_path = os.path.join(root, file)
                npz_path = pt_path.replace('.pt', '.npz')
                try:
                    tensor = torch.load(pt_path)
                    array = tensor.cpu().numpy()
                    np.savez_compressed(npz_path, data=array)
                    os.remove(pt_path)
                    logging.info(f"Converted and deleted: {pt_path}")
                    count += 1
                except Exception as e:
                    logging.error(f"Failed to convert {pt_path}: {e}")

    total_time = time.time() - start_time
    logging.info(f"Total files processed: {count}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt files to .npz recursively")
    parser.add_argument("input_dir", type=str, help="Path to directory containing .pt files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    convert_pt_to_npz(args.input_dir)

