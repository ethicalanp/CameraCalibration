import os

def strip_pose_labels(label_dir, output_dir):
    """
    Process label files in label_dir, strip pose data (from 35 to 29 values),
    and save modified files to output_dir.
    """
    total_files = 0
    modified_files = 0
    skipped_files = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                # Construct output filepath
                rel_path = os.path.relpath(filepath, label_dir)
                output_filepath = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    new_lines = []
                    valid_file = True

                    for line in lines:
                        parts = line.strip().split()

                        if len(parts) == 35:
                            parts = parts[:29]
                            new_lines.append(" ".join(parts) + "\n")
                        elif len(parts) == 29:
                            new_lines.append(line)
                        else:
                            skipped_files.append(f"{filepath} - {len(parts)} values")
                            valid_file = False
                            break

                    if valid_file:
                        with open(output_filepath, 'w') as f_out:
                            f_out.writelines(new_lines)
                        modified_files += 1
                    total_files += 1

    print(f"\n✅ Completed: {modified_files} files fixed out of {total_files} total label files.")
    print(f"⚠️ Skipped {len(skipped_files)} files with unexpected format.")

    if skipped_files:
        skip_log_path = os.path.join(output_dir, "skipped_files.txt")
        with open(skip_log_path, "w") as skip_log:
            skip_log.write("\n".join(skipped_files))
        print(f"📄 Skipped file list saved to: {skip_log_path}")

if __name__ == "__main__":
    label_folder = r"C:\Users\ACER\Documents\PROJECT\Convert_to_Linemod\Step 2\input"  # Input labels
    output_folder = r"C:\Users\ACER\Documents\PROJECT\Convert_to_Linemod\Step 2\output"  # Output labels
    strip_pose_labels(label_folder, output_folder)