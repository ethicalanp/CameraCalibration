import os
import random

def split_dataset(images_path, output_dir,
                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    # Collect all PNG files
    images = [f for f in os.listdir(images_path) if f.endswith('.png')]
    images.sort()
    random.shuffle(images)

    total = len(images)
    train = images[:int(train_ratio * total)]
    val = images[int(train_ratio * total):int((train_ratio + val_ratio) * total)]
    test = images[int((train_ratio + val_ratio) * total):]

    os.makedirs(output_dir, exist_ok=True)

    def write_list(name, data):
        file_path = os.path.join(output_dir, f"{name}.txt")
        with open(file_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")   # only filename
        print(f"✅ Saved {len(data)} items to {file_path}")
        return data, file_path

    # Write train/val/test files
    train, train_file = write_list("train", train)
    val, val_file = write_list("val", val)
    test, test_file = write_list("test", test)

    # Create training_range.txt from train set (only numbers)
    training_range_path = os.path.join(output_dir, "training_range.txt")
    with open(training_range_path, 'w') as f:
        for item in train:
            # Extract just the number from frame_XXXX.png
            number = os.path.splitext(item)[0].replace("frame_", "")
            f.write(f"{number}\n")
    print(f"✅ Saved training range ({len(train)} numbers) to {training_range_path}")


if __name__ == "__main__":
    base_dir = r"C:\Users\ACER\Documents\PROJECT\Convert_to_Linemod\Step 3"
    images_path = os.path.join(base_dir, "inputs", "PNGImages")
    output_dir = os.path.join(base_dir, "output")

    split_dataset(images_path, output_dir)