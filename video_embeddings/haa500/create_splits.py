import os
import argparse

def generate_splits(input_folder, output_folder):
    train_lines = []
    val_lines = []
    test_lines = []

    for file in os.listdir(input_folder):
        if file.endswith(".txt"):
            class_name = os.path.splitext(file)[0]

            # Generate 20 sample IDs for this class
            sample_ids = [f"{class_name}_{i:03d}" for i in range(20)]

            # Split
            train_lines.extend(sample_ids[:16])
            val_lines.append(sample_ids[16])
            test_lines.extend(sample_ids[17:])

    # Make sure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Write files
    with open(os.path.join(output_folder, "train.txt"), "w") as f:
        f.write("\n".join(train_lines))

    with open(os.path.join(output_folder, "val.txt"), "w") as f:
        f.write("\n".join(val_lines))

    with open(os.path.join(output_folder, "test.txt"), "w") as f:
        f.write("\n".join(test_lines))

    print("✅ train.txt, val.txt, test.txt created in", output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits for HAA500-style dataset")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder containing class .txt files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to save train.txt, val.txt, test.txt")

    args = parser.parse_args()
    generate_splits(args.input_folder, args.output_folder)
