# SLAM Project Explanation: Transition to Monocular Python SLAM

This document explains the architecture, dependencies, and implementation of this project. 

Originally, this repository was configured to perform 3D SLAM using two physical Intel RealSense cameras (D435i and T265) integrated via ROS Melodic on a TurtleBot 2 robot. We transitioned the project into a **standalone, pure-Python monocular SLAM system** that works on Windows/Linux/Mac out-of-the-box using any webcam, video file, or drone feed (like a DJI O4 stream).

---

## 1. Project Evolution: Hardware ROS vs. Monocular Python

| Feature | Original Setup | New Setup |
| :--- | :--- | :--- |
| **Hardware Required** | Intel RealSense D435i + T265 | Any standard Monocular Camera (webcam, video, or stream) |
| **Software Platform** | Ubuntu 18.04 + ROS Melodic | Cross-Platform (Windows/Mac/Linux) + Pure Python |
| **3D Mapping Source** | D435i Hardware Depth Sensor | **Depth Anything V2** (Monocular Deep Learning Depth) |
| **Tracking Source** | T265 Visual-Inertial Hardware Sensor | **Custom Python Visual Odometry** (OpenCV LK Optical Flow) |
| **3D Visualization** | ROS RViz | **Open3D** (Interactive OpenGL/GLFW Visualizer) |
| **Compilation** | Heavy C++ builds (Catkin) | **Zero Compilation** (pip install libraries only) |

---

## 2. Core Technologies & Libraries Used

*   **Python 3:** The primary programming language.
*   **PyTorch & Hugging Face Transformers (`torch`, `transformers`):** Used to load and execute the pre-trained neural network **Depth Anything V2 Small** for zero-shot relative depth estimation.
*   **OpenCV (`opencv-python`):** Handles:
    *   Reading video streams (`cv2.VideoCapture`).
    *   Image preprocessing and Grayscale conversion.
    *   Fast corner feature detection (`cv2.FastFeatureDetector`).
    *   Lucas-Kanade Sparse Optical Flow point tracking (`cv2.calcOpticalFlowPyrLK`).
    *   Relative translation and rotation estimation (`cv2.findEssentialMat` and `cv2.recoverPose` with RANSAC outlier rejection).
    *   Split-screen visual dashboard rendering.
*   **Open3D (`open3d`):** A modern, lightweight 3D graphics library. Used to render:
    *   Coordinate Axes (Red = X, Green = Y, Blue = Z).
    *   Camera Trajectory (3D red path line).
    *   Global Point Cloud Map (color-coded 3D sparse edge outlines).
*   **NumPy (`numpy`):** Handles high-performance mathematical operations, matrix coordinate projection, and coordinate transformations ($P_{\text{world}} = R \cdot P_{\text{camera}} + t$).

---

## 3. Storage & Environment Configuration

To accommodate systems with limited space on drive `C:`, we isolated the setup to drive `D:`:
1.  **D-Drive Virtual Environment (`venv`):** All python packages (including PyTorch and Open3D, totaling ~970 MB) are installed in the local `venv` directory on the D drive.
2.  **D-Drive Model Cache (`.cache/`):** We programmatically override Hugging Face and PyTorch cache directories inside the Python scripts:
    *   `os.environ["HF_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/huggingface"`
    *   `os.environ["TORCH_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/torch"`
    This redirects the ~95 MB model weight downloads completely to your D drive, keeping your C drive clean.

---

## 4. Script Breakdown

### Phase 1: Depth Estimation & Sparsing
*   **`scripts/depth_estimator.py`:** Wrapper class for Depth Anything V2. Loads model weights on CPU/GPU and generates 2D depth predictions.
*   **`scripts/test_on_image.py`:** Reads a single frame (webcam or `test.jpg`), predicts depth, runs Canny edge detection to isolate structural contours, back-projects the 2D edge pixels into a 3D point cloud using camera intrinsics, and displays the static 3D point cloud in Open3D.

### Phase 2: Visual Odometry Tracking
*   **`scripts/visual_odometry.py`:** The visual tracking engine. Tracks feature corners frame-to-frame and calculates relative camera rotation ($R$) and translation direction ($t$) using essential matrix decomposition.
*   **`scripts/test_vo.py`:** An odometry runner. Stacks the live color feed (left) and grayscale tracking feed (center) into a single OpenCV window, and opens a dynamic Open3D window on the right drawing the camera path in real-time.

### Phase 3: Fused SLAM & Mapping
*   **`scripts/mono_slam.py`:** The unified SLAM program. Combines depth prediction, camera path tracking, Canny edge contour detection, 3D point coordinate projection, and **Voxel Downsampling** (merging duplicate 3D points in a 5 cm grid to prevent memory bloat). Renders the split video dashboard and the growing 3D sparse edge map of the room concurrently.
