import cv2
import cv2.aruco as Aruco
import numpy as np
import os
import json

# -------------------- USER CONFIG --------------------
video_path = "C:/Users/ACER/Documents/PROJECT/02-Calibration/Inputs/Charuco_board.mp4"
output_path = "C:/Users/ACER/Documents/PROJECT/02-Calibration/Outputs/camera_params.json"
square_size = 0.021 + 0.0025  # Total size of one square (marker_length + marker_separation)
pattern_size = (4, 5)  # Adjusted for ~20 corners, closer to 24-marker border
aruco_dict_type = Aruco.DICT_6X6_50

# Initialize ArUco dictionary
aruco_dict = Aruco.getPredefinedDictionary(aruco_dict_type)

# Prepare object points (3D points in real world space)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all frames
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

decimator = 0
image_size = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    if img is None or img.size == 0:
        print(f"[WARNING] Frame {decimator} is invalid, skipping.")
        decimator += 1
        continue

    # Process every frame for maximum diversity
    if decimator % 1 == 0:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = Aruco.detectMarkers(gray, aruco_dict)

            print(f"[DEBUG] Frame {decimator}: Detected {len(ids) if ids is not None else 0} markers, {len(corners) if corners else 0} corners.")

            if ids is not None and len(ids) > 5:
                if len(corners) > 10:
                    try:
                        corners_flat = np.array([item for sublist in corners for item in sublist]).reshape(-1, 2)
                        if corners_flat.shape[0] >= 20:  # Accept 20+ corners
                            # Filter outliers by checking corner consistency
                            if len(np.unique(corners_flat, axis=0)) >= 0.8 * len(corners_flat):  # 80% unique
                                imgpoints.append(corners_flat[:pattern_size[0] * pattern_size[1]])  # Use up to 20 points
                                objpoints.append(objp)
                                print(f"[DEBUG] Frame {decimator}: Added {len(corners_flat[:pattern_size[0] * pattern_size[1]])} corners to calibration.")
                                # Draw detected markers for visualization
                                img_with_markers = img.copy()
                                Aruco.drawDetectedMarkers(img_with_markers, corners, ids)
                                cv2.imshow('Detected ArUco', img_with_markers)
                                cv2.waitKey(1)  # Brief display
                    except ValueError as e:
                        print(f"[WARNING] Frame {decimator}: Error reshaping corners - {e}, skipping frame.")
                        continue
            else:
                print(f"[DEBUG] Frame {decimator}: Insufficient markers ({len(ids) if ids is not None else 0}), skipping.")
        except Exception as e:
            print(f"[ERROR] Frame {decimator}: Processing failed - {e}, skipping frame.")
            continue

        # Capture image size from the first valid frame
        if image_size is None and img is not None:
            image_size = (img.shape[1], img.shape[0])  # width, height

    decimator += 1

cv2.destroyAllWindows()
cap.release()

if not objpoints:
    print("Error: No valid frames with sufficient ArUco corners detected. Check video or use a 24-marker border ArUco board.")
    print("Debug: Ensure the video contains a 24-marker border ArUco board with DICT_6X6_50, fully visible, and with good lighting.")
    exit(1)

if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
    print("Error: Invalid image size detected. Check video file.")
    exit(1)

# Perform camera calibration
print(f"[INFO] Calibrating with {len(objpoints)} frames...")
ret, mtx, dtx, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
# Validate reprojection error
if ret > 5.0:  # Relaxed threshold
    print(f"[ERROR] Reprojection error {ret} is too high. Calibration may be unreliable. Check video or board setup.")
    exit(1)
print(f"[DEBUG] Calibration reprojection error: {ret}")
print(f"[DEBUG] Intrinsic matrix:\n{mtx}")
print(f"[DEBUG] Distortion coefficients:\n{dtx}")

# Save camera parameters to JSON
camera_params = {
    "intrinsic": mtx.tolist(),
    "distortion": dtx.tolist(),
    "reprojection_error": float(ret)
}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as outfile:
    json.dump(camera_params, outfile, indent=4)
print(f"[INFO] Camera parameters saved to {output_path}")

# Optional: Test undistortion
test_image_path = "/home/pravneeth/Desktop/AI4SEE/sample.jpeg"
if os.path.exists(test_image_path):
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dtx, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dtx, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)
    print("[INFO] Undistorted image saved as calibresult.png")
else:
    print("[WARNING] Test image not found, skipping undistortion test.")