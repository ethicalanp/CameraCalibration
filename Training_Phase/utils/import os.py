import os

# Paths to your split text files
txt_files = [
    "/home/pravneeth/Desktop/PROJECT/05-Training_Phase/data/LINEMOD/Mouse/train.txt",
    "/home/pravneeth/Desktop/PROJECT/05-Training_Phase/data/LINEMOD/Mouse/val.txt",
    "/home/pravneeth/Desktop/PROJECT/05-Training_Phase/data/LINEMOD/Mouse/test.txt",
]

def truncate_paths(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Keep only the last part after "/"
    new_lines = [os.path.basename(line.strip()) + "\n" for line in lines if line.strip()]
    
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated: {file_path}")

for txt in txt_files:
    truncate_paths(txt)
