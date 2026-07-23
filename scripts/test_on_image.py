import os
# Force Hugging Face and Torch to download and cache models on the D drive
os.environ["HF_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/huggingface"
os.environ["TORCH_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/torch"

import cv2
import numpy as np
import open3d as o3d
import sys
from depth_estimator import DepthEstimator

def project_depth_to_3d_edges(frame, depth_map, fx=500.0, fy=500.0, cx=320.0, cy=240.0, low_threshold=50, high_threshold=150):
    """
    Detects edges in the frame, masks the depth map to only contain values at the edges,
    and projects the 2D edge pixel coordinates into 3D camera coordinates.
    """
    # 1. Convert to grayscale and detect Canny edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Save the edge image for user reference
    cv2.imwrite("detected_edges.jpg", edges)
    print("[Pipeline] Saved Canny edge detection mask as 'detected_edges.jpg'")

    # 2. Get the indices of the edge pixels
    edge_y, edge_x = np.where(edges > 0)
    
    if len(edge_x) == 0:
        print("[Pipeline] Error: No edges detected in this frame.")
        return None
    
    # 3. Extract the depth at these edge pixels
    # Depth Anything V2 outputs relative depth values.
    # In some models, the output is inverse depth (disparity). Let's extract values directly first.
    depths = depth_map[edge_y, edge_x]
    
    # We clip extreme values to prevent mathematical instability
    depths = np.clip(depths, 0.1, 100.0)
    
    # 4. Project coordinates using the pinhole camera model equations:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    Z = depths
    X = (edge_x - cx) * Z / fx
    Y = (edge_y - cy) * Z / fy
    
    # Stack coordinates to form (N, 3) points array
    points = np.stack((X, Y, Z), axis=-1)
    
    # Extract corresponding RGB colors for each point (scale to [0, 1] for Open3D)
    colors = frame[edge_y, edge_x][:, ::-1] / 255.0
    
    return points, colors

def main():
    # 1. Initialize Depth Estimator
    estimator = DepthEstimator()
    
    # 2. Choose input image
    image_path = "test.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    if os.path.exists(image_path):
        print(f"[Pipeline] Loading image from path: '{image_path}'")
        frame = cv2.imread(image_path)
    else:
        print(f"[Pipeline] Image '{image_path}' not found in directory.")
        print("[Pipeline] Attempting to capture a test frame from default webcam (device 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Pipeline] Error: Could not open default webcam.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[Pipeline] Error: Could not capture frame from webcam.")
            return
        # Save captured frame as test.jpg for reuse
        cv2.imwrite("test.jpg", frame)
        print("[Pipeline] Captured and saved frame as 'test.jpg'.")

    h, w, c = frame.shape
    print(f"[Pipeline] Resolution: {w}x{h}")
    
    # 3. Predict Depth Map
    print("[Pipeline] Running Depth Anything V2 inference...")
    depth_map = estimator.predict(frame)
    print(f"[Pipeline] Raw Depth range - Min: {depth_map.min():.4f}, Max: {depth_map.max():.4f}")
    
    # Normalize depth map for visual saving
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite("depth_map.jpg", depth_colormap)
    print("[Pipeline] Saved depth visualization as 'depth_map.jpg'")
    
    # 4. Project to 3D Points
    print("[Pipeline] Projecting edge depth to 3D point cloud...")
    # Default intrinsic camera settings (can be calibrated later)
    fx, fy = 500.0, 500.0
    cx, cy = w / 2.0, h / 2.0
    
    proj_result = project_depth_to_3d_edges(frame, depth_map, fx, fy, cx, cy)
    if proj_result is None:
        return
    
    points, colors = proj_result
    print(f"[Pipeline] Successfully projected {len(points)} points to 3D.")
    
    # 5. Visualize in Open3D
    print("[Pipeline] Initializing Open3D Visualizer...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Display the point cloud window
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name="3D Sparse Edge Point Cloud", 
        width=1024, 
        height=768
    )
    print("[Pipeline] Visualization complete.")

if __name__ == "__main__":
    main()
