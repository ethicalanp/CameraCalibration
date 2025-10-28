import os

# path to the txt directory
txt_dir = "data/LINEMOD/Mouse"

for txt_file in ["train.txt", "val.txt", "test.txt"]:
    file_path = os.path.join(txt_dir, txt_file)
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping missing file: {file_path}")
        continue

    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    # keep only the last part after "/" (filename only)
    new_lines = [os.path.basename(line.strip()) for line in lines if line.strip()]

    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✅ Fixed {txt_file} → now contains {len(new_lines)} entries (only filenames)")
