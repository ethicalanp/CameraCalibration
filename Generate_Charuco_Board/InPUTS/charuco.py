import cv2
import cv2.aruco as aruco
import numpy as np

# ----- CONFIGURE BOARD HERE -----
squares_x = 8  # columns (X axis)
squares_y = 5  # rows (Y axis)
square_length = 0.035  # meters
marker_length = 0.013  # meters
dictionary_id = aruco.DICT_4X4_50

# A4 size in mm
a4_width_mm = 210
a4_height_mm = 297
dpi = 300

# Convert A4 size to pixels
a4_width_px = int(a4_width_mm / 25.4 * dpi)
a4_height_px = int(a4_height_mm / 25.4 * dpi)

# ----- GENERATE CHARUCO BOARD -----
try:
    # Load the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_id)
    
    # Create the Charuco board
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    # Generate the board image
    img = board.generateImage((a4_width_px, a4_height_px), marginSize=20, borderBits=1)
    
    # Save the image
    cv2.imwrite("charuco_8x5_A4.png", img)
    print("[INFO] Charuco board saved as 'charuco_8x5_A4.png'")
except Exception as e:
    print(f"[ERROR] Failed to generate or save Charuco board: {e}")