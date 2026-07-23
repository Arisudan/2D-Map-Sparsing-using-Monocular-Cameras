import os
# Force Hugging Face and Torch to download and cache models on the D drive
os.environ["HF_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/huggingface"
os.environ["TORCH_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/torch"

import cv2
import numpy as np
import sys
import open3d as o3d
import datetime
from depth_estimator import DepthEstimator
from visual_odometry import VisualOdometry

def main():
    # Camera Intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0

    # 1. Initialize SLAM Modules
    print("[SLAM] Initializing Depth Anything V2...")
    depth_estimator = DepthEstimator()
    print("[SLAM] Initializing Visual Odometry...")
    vo = VisualOdometry(fx=fx, fy=fy, cx=cx, cy=cy)

    # 2. Setup video source (Webcam, MP4 file, or stream URL)
    video_source = 0  # Default to default webcam (device index 0)
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        if source_arg.isdigit():
            video_source = int(source_arg)
        elif os.path.exists(source_arg) or source_arg.startswith("rtsp://") or source_arg.startswith("rtmp://"):
            video_source = source_arg
            print(f"[SLAM] Loading video source: {video_source}")
        else:
            print(f"[SLAM] Error: Video source '{source_arg}' not found.")
            return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[SLAM] Error: Could not open video source '{video_source}'")
        return

    print("[SLAM] Video source opened successfully.")
    print("[SLAM] Press 'q' or 'ESC' inside the dashboard window to quit.")

    # 3. Initialize Open3D Visualizer for 3D Sparse Edge Mapping on the right
    print("[SLAM] Initializing Open3D 3D Map Visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Sparse Edge SLAM Map (Open3D)", width=600, height=600, left=650, top=50)

    # Add Coordinate Axis Helper at Origin
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis_frame)

    # Add Trajectory LineSet
    path_points = [np.array([0.0, 0.0, 0.0])]
    path_line_set = o3d.geometry.LineSet()
    path_line_set.points = o3d.utility.Vector3dVector(np.array(path_points))
    vis.add_geometry(path_line_set)

    # Add Global Point Cloud Map
    map_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(map_pcd)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[SLAM] Video source finished or disconnected.")
            break

        frame_count += 1
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # --- STEP 1: Run Visual Odometry Tracking ---
        R, t = vo.process_frame(frame)
        tx, ty, tz = t[0, 0], t[1, 0], t[2, 0]
        
        # Append to camera path (negating Y/Z for Open3D standard viewport)
        path_points.append(np.array([tx, -ty, -tz]))

        # --- STEP 2: Run Depth Inference ---
        depth_map = depth_estimator.predict(frame)
        
        # Normalize relative depth from Depth Anything V2 to metric scale
        # We assume an indoor room median depth scale of 2.0 meters
        median_depth = np.median(depth_map)
        scaled_depth = depth_map * (2.0 / (median_depth + 1e-6))
        scaled_depth = np.clip(scaled_depth, 0.1, 10.0)

        # --- STEP 3: Detect Canny Edges ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_y, edge_x = np.where(edges > 0)

        # Subsample edge points to avoid lag (limit to 1000 points per frame)
        if len(edge_x) > 1000:
            indices = np.random.choice(len(edge_x), size=1000, replace=False)
            edge_x = edge_x[indices]
            edge_y = edge_y[indices]

        # --- STEP 4: Project and Transform Edge Points to 3D ---
        if len(edge_x) > 0:
            Z = scaled_depth[edge_y, edge_x]
            X = (edge_x - cx) * Z / fx
            Y = (edge_y - cy) * Z / fy
            
            # 3D points in camera coordinates (ROS frame: X right, Y down, Z forward)
            # Open3D coordinate frame (X right, Y up, Z backward) needs Y/Z negated
            pts_camera = np.stack((X, -Y, -Z), axis=-1)
            
            # Transform points from camera frame to global world coordinate frame
            # World coordinates = Rotation_global * Camera_points + Translation_global
            # (Note: we negate translation Y/Z coordinates to match the path coordinate frame)
            t_world = np.array([tx, -ty, -tz])
            pts_world = pts_camera.dot(R.T) + t_world
            
            # Extract point colors from original color frame
            colors = frame[edge_y, edge_x][:, ::-1] / 255.0

            # Accumulate into global Point Cloud
            old_pts = np.asarray(map_pcd.points)
            old_cols = np.asarray(map_pcd.colors)
            
            if len(old_pts) > 0:
                merged_pts = np.vstack((old_pts, pts_world))
                merged_cols = np.vstack((old_cols, colors))
            else:
                merged_pts = pts_world
                merged_cols = colors
                
            map_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            map_pcd.colors = o3d.utility.Vector3dVector(merged_cols)

            # Voxel downsampling (combines duplicates within 5cm voxels)
            # This maintains a sparse wireframe and keeps rendering fast
            map_pcd = map_pcd.voxel_down_sample(voxel_size=0.05)

        # --- STEP 5: Update Open3D Renderer ---
        pts_arr = np.array(path_points)
        num_pts = len(pts_arr)
        
        # Update path line set geometry
        if num_pts > 1:
            lines = [[i, i + 1] for i in range(num_pts - 1)]
            path_colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]  # Red path line
            
            path_line_set.points = o3d.utility.Vector3dVector(pts_arr)
            path_line_set.lines = o3d.utility.Vector2iVector(lines)
            path_line_set.colors = o3d.utility.Vector3dVector(path_colors)
            
            # Refresh trajectory line
            vis.remove_geometry(path_line_set, reset_bounding_box=False)
            vis.add_geometry(path_line_set, reset_bounding_box=False)

        # Refresh map point cloud
        vis.remove_geometry(map_pcd, reset_bounding_box=False)
        vis.add_geometry(map_pcd, reset_bounding_box=False)

        # Trigger viewport auto-focus once at the beginning
        if frame_count == 10:
            vis.reset_view_point(True)

        vis.poll_events()
        vis.update_renderer()

        # --- STEP 6: Render OpenCV Split Dashboard (Color | Grayscale Edges) ---
        color_dashboard = frame.copy()
        # Draw green tracking points on color feed
        if vo.prev_pts is not None:
            for pt in vo.prev_pts:
                cv2.circle(color_dashboard, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # Overlay coordinates
        pos_text = f"X: {tx:.2f}, Y: {ty:.2f}, Z: {tz:.2f}"
        cv2.putText(color_dashboard, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(color_dashboard, f"Tracked Pts: {len(vo.prev_pts) if vo.prev_pts is not None else 0}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_dashboard, "LIVE COLOR FEED", (20, color_dashboard.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Prepare center grayscale edges dashboard
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(edges_bgr, "GRAYSCALE EDGES", (20, edges_bgr.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Horizontal stack dashboard
        dashboard = np.hstack((color_dashboard, edges_bgr))
        cv2.imshow("SLAM Dashboard (Color | Grayscale Edges)", dashboard)

        # Listen for quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[SLAM] Trajectory tracking stopped.")

    # Save the point cloud map to the maps folder on exit
    if len(map_pcd.points) > 0:
        os.makedirs("maps", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maps/Map1_{timestamp}.ply"
        print(f"[SLAM] Saving accumulated 3D map to '{filename}'...")
        success = o3d.io.write_point_cloud(filename, map_pcd)
        if success:
            print("[SLAM] Map saved successfully.")
        else:
            print("[SLAM] Error: Failed to save map.")
    else:
        print("[SLAM] Map is empty. Nothing to save.")

if __name__ == "__main__":
    main()
