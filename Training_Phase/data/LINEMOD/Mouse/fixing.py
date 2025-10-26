import glob

# all txt files you want to fix
txt_files = glob.glob("data/LINEMOD/Mouse/*.txt")

for txt in txt_files:
    with open(txt, "r") as f:
        lines = f.readlines()
    # fix paths
    new_lines = [line.replace("JPEGImages/JPEGImages", "JPEGImages") for line in lines]
    with open(txt, "w") as f:
        f.writelines(new_lines)
    print(f"✅ Fixed {txt}")

