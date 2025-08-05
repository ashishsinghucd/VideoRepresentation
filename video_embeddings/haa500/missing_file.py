import os
from collections import defaultdict

def find_missing_filenames(folder_path, expected_count=20):
    files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    base_to_indices = defaultdict(set)

    for f in files:
        if '_' not in f:
            continue
        base, idx = f.rsplit('_', 1)
        idx = idx.replace('.pt', '')
        if idx.isdigit():
            base_to_indices[base].add(int(idx))

    missing_full_filenames = []
    for base, indices in base_to_indices.items():
        for i in range(expected_count):
            if i not in indices:
                missing_full_filenames.append(f"{base}_{i:03d}.pt")

    return missing_full_filenames

# Replace this with your folder path
folder = "/home/ashisig/Research/Data/HAA500/video_embeddings"
missing = find_missing_filenames(folder)

# Print as comma-separated string list with double quotes
formatted_list = ', '.join(f'"{m}"' for m in missing)
print(f"[{formatted_list}]")

