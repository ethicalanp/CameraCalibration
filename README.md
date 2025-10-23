CameraCalibration
A Python-based camera calibration system using OpenCV and ArUco markers to generate a Charuco board for precise intrinsic and extrinsic camera parameter estimation, optimized for computer vision applications.
Overview
This project generates a Charuco board (a hybrid chessboard-ArUco marker pattern) for camera calibration. The board is designed for A4 printing and used to calibrate cameras by capturing images from multiple angles. The project leverages OpenCV’s ArUco module and supports CUDA acceleration for potential high-performance image processing. It is ideal for applications like robotics, augmented reality, and 3D reconstruction.
Features

Generates an 8x5 Charuco board with customizable square (0.0345m) and marker (0.021m) sizes.
Outputs a high-resolution A4-sized PNG (210x297mm at 300 DPI) for printing.
Configurable ArUco dictionary (default: DICT_6X6_50) for robust marker detection.
Compatible with CUDA-enabled OpenCV for GPU-accelerated processing (optional).
Organized directory structure for easy integration into larger vision pipelines.

Prerequisites

Python: 3.8 or higher
OpenCV: 4.5.0 or higher (with opencv-contrib-python for ArUco module)
NumPy: 1.21.0 or higher
CUDA Toolkit: 12.1 (optional, for GPU acceleration)
NVIDIA GPU and drivers: Compatible with CUDA 12.1 (e.g., driver version 525+)
Operating System: Tested on Ubuntu 20.04/22.04

Installation

Clone the repository:
git clone https://github.com/yourusername/CameraCalibration.git
cd CameraCalibration


Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate


Install dependencies:
pip install opencv-contrib-python numpy
pip install requirements.txt

Verify CUDA (optional):

Ensure CUDA 12.1 is installed: nvcc --version.
Confirm OpenCV CUDA support: python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())".
If needed, rebuild OpenCV with CUDA: Follow OpenCV CUDA build guide.



Usage

Generate the Charuco board:
python generate_charuco_board.py


This creates output/charuco_8x5_A4.png.
Print the PNG on an A4 sheet for calibration.


Calibrate a camera (example, not included in repo):

Capture images of the printed board from multiple angles.
Use OpenCV’s cv2.aruco functions to detect markers and corners, then calibrate with cv2.aruco.calibrateCameraCharuco.
See OpenCV’s camera calibration tutorial for details.



Directory Structure
CameraCalibration/
├── generate_charuco_board.py  # Script to generate the Charuco board
├── output/                   # Directory for generated PNG
├── README.md                 # Project documentation

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
