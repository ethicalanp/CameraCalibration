import sys, os
import cv2
import numpy as np
import pose_utils
import json
import matplotlib.pyplot as plt
from PIL import Image

# -------------------- USER CONFIG --------------------
object_path = "C:/Users/ACER/Documents/PROJECT/03-Create_Dataset/INPUTS/Mouse2.ply"
camera_path = "C:/Users/ACER/Documents/PROJECT/03-Create_Dataset/INPUTS/camera_params.json"
data_path = "C:/Users/ACER/Documents/PROJECT/03-Create_Dataset/INPUTS/Object.mp4"
output_path = "C:/Users/ACER/Documents/PROJECT/03-Create_Dataset/OUTPUTS"

# load 3D model
mesh = pose_utils.MeshPly(object_path)
print(f"[DEBUG] Raw vertex range (min, max): {np.min(mesh.vertices, axis=0)}, {np.max(mesh.vertices, axis=0)}")
vertices_og = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose() # Remove /1000
print(f"[DEBUG] Unscaled vertex range (min, max): {np.min(vertices_og[:3], axis=1)}, {np.max(vertices_og[:3], axis=1)}")
corners3D = pose_utils.get_3D_corners(vertices_og)
vertices = np.hstack((np.array([0,0,0,1]).reshape(4,1), corners3D)) # add center coordinate

# load camera params
with open(camera_path, 'r') as f:
    camera_data = json.load(f)

# camera_model = None
dtx = np.array(camera_data["distortion"])
mtx = np.array(camera_data["intrinsic"])

# Predefined offset from charuco frame to object frame
rotation_offset = [180.0, 180.0, -90.0] # degrees
translation_offset = [0.145, 0.095, 0.0] # translation in meters #  [0.221, 0.14075, -0.0485 ]
# 0.145, 0.96, 0.0


offset_mat = pose_utils.construct_transform(translation_offset, rotation_offset)

# 1. Run through video
cap = cv2.VideoCapture(data_path)
frame_count = 0
# max_frame_count = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # if frame_count > max_frame_count:
    #     print("Max frame count reached")
    #     break

    print(f"Frame {frame_count}")
    # 2. Detect ChArUco pose
    frame_remapped_gray = frame[:, :, 0]
    im_height, im_width = frame_remapped_gray.shape
    im_with_charuco_board, pose = pose_utils.detect_Charuco_pose_board(frame, mtx, dtx, markerSize=6, totalMarkers=50,
                                                                      number_markers_x=8, number_markers_y=5,
                                                                      square_width=0.034, aruco_width=0.021)
    if pose is not None:
      # 3. determine offset from charuco pose to object pose
      rvec = pose['rvec']
      tvec = pose['tvec']
      rotation = cv2.Rodrigues(rvec)[0]
      transform_mat = np.vstack((np.hstack((rotation, tvec)), np.array([0, 0, 0, 1])))
      transform_mat = np.matmul(transform_mat, offset_mat)
      print(f"[DEBUG] Transform matrix:\n{transform_mat}")
      # 4. project object onto image
      projected_corners = pose_utils.compute_projection(corners3D, transform_mat[:3, :], mtx)
      projected_vertices = pose_utils.compute_projection(vertices, transform_mat[:3, :], mtx)
      print(f"[DEBUG] Projected vertices shape: {projected_vertices.shape}")
      print(f"[DEBUG] Projected vertices: {projected_vertices}")
      print(f"[DEBUG] Projected corners shape: {projected_corners.shape}")
      print(f"[DEBUG] Projected corners: {projected_corners}")
      for x, y in projected_vertices.T:
          if not (0 <= x < im_width and 0 <= y < im_height):
              print(f"[WARNING] Point ({x}, {y}) outside image bounds [{im_width}, {im_height}]")
      # 5. Draw projected object
      im_with_charuco_board = pose_utils.draw_BBox(im_with_charuco_board, projected_corners.T, projected_vertices.T)
      # Create mask using full mesh
      mask_arr = pose_utils.create_mask_from_mesh(vertices_og, mesh.indices, transform_mat, mtx, im_width, im_height)
      if mask_arr is None or cv2.countNonZero(mask_arr) == 0:
          print("[WARNING] Mask from mesh failed, falling back to simple mask.")
          mask_arr = pose_utils.create_simple_mask(projected_vertices.T, im_width, im_height)
      print(f"[DEBUG] Mask non-zero pixels: {cv2.countNonZero(mask_arr)}")
      # Create label
      label = pose_utils.create_label(0, projected_vertices, mtx[0,0], mtx[1,1], im_width, im_height, mtx[0,2], mtx[1,2], im_width, im_height, transform_mat)
      imageName = f"frame_{frame_count}.png"
      # 6. store all information
      pose_utils.save_data(frame, mask_arr, label, imageName, output_path, im_with_charuco_board)

    frame_count += 1

cap.release()

# Visualization
img_path = os.path.join(output_path, "images")
if not os.path.isdir(img_path) or not any(file.endswith(".png") for file in os.listdir(img_path)):
    print("No images were saved. Check ChArUco board detection.")
else:
    label_path = os.path.join(output_path, "labels")
    mask_path = os.path.join(output_path, "mask")

    for file in os.listdir(img_path):
        if file.endswith(".png"):
            assert os.path.isfile(os.path.join(label_path, file[:-4] + ".txt")), f"Label file {file[:-4] + '.txt'} does not exist"
            assert os.path.isfile(os.path.join(mask_path, file)), f"Mask file {file} does not exist"

            img = cv2.imread(os.path.join(img_path, file))
            label = np.loadtxt(os.path.join(label_path, file[:-4] + ".txt"))
            mask = cv2.imread(os.path.join(mask_path, file))
            break

    keypoints = label[1:19]
    keypoints_x = keypoints[::2] * img.shape[1]
    keypoints_y = keypoints[1::2] * img.shape[0]

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.scatter(keypoints_x, keypoints_y, s=10, c='red', marker='o')
    plt.show()