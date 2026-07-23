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
    # Using the HEAVY Depth Anything V2 Large model for precise structural boundaries
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"[SLAM] Initializing Heavy Depth Anything V2 ({model_name})...")
    depth_estimator = DepthEstimator(model_name=model_name)
    
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
    vis.create_window(window_name="3D Sparse Edge SLAM Map (Open3D)", width=600, height=600, left=820, top=50)

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

    # 4. Initialize 2D Floor Plan Grid Map
    # 480x480 canvas representing 9.6m x 9.6m area (scale: 50 pixels per meter)
    grid_map_size = 480
    grid_map = np.zeros((grid_map_size, grid_map_size), dtype=np.uint8)
    draw_scale = 50.0
    center_grid = grid_map_size // 2

    # Optimization parameters
    frame_count = 0
    depth_interval = 10  # Only run heavy depth estimator once every 10 frames
    latest_depth_map = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[SLAM] Video source finished or disconnected.")
            break

        frame_count += 1
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # --- STEP 1: Run Visual Odometry Tracking (Runs at 30 FPS) ---
        R, t = vo.process_frame(frame)
        tx, ty, tz = t[0, 0], t[1, 0], t[2, 0]
        
        # Append to camera path (negating Y/Z for Open3D standard viewport)
        path_points.append(np.array([tx, -ty, -tz]))

        # --- STEP 2: Keyframe Filter for Heavy Depth Inference ---
        # Run Depth Anything V2 only on keyframes to eliminate latency
        is_keyframe = (frame_count % depth_interval == 1) or (latest_depth_map is None)
        
        if is_keyframe:
            # Predict relative depth map
            raw_depth = depth_estimator.predict(frame)
            # Normalize to metric scale assuming an indoor median-depth of 2.0 meters
            median_depth = np.median(raw_depth)
            scaled_depth = raw_depth * (2.0 / (median_depth + 1e-6))
            latest_depth_map = np.clip(scaled_depth, 0.1, 10.0)

        # --- STEP 3: Detect Canny Edges (Room Boundary Outlines) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # --- STEP 4: Project and Transform Edge Points to 3D and 2D ---
        if is_keyframe and latest_depth_map is not None:
            edge_y, edge_x = np.where(edges > 0)
            
            # Subsample edge points to avoid rendering lag (limit to 1000 points per keyframe)
            if len(edge_x) > 1000:
                indices = np.random.choice(len(edge_x), size=1000, replace=False)
                edge_x = edge_x[indices]
                edge_y = edge_y[indices]

            if len(edge_x) > 0:
                Z = latest_depth_map[edge_y, edge_x]
                X = (edge_x - cx) * Z / fx
                Y = (edge_y - cy) * Z / fy
                
                # 3D coordinates in Open3D frame (Y and Z negated)
                pts_camera = np.stack((X, -Y, -Z), axis=-1)
                
                # Transform camera points to world frame
                t_world = np.array([tx, -ty, -tz])
                pts_world = pts_camera.dot(R.T) + t_world
                
                # Extract RGB point colors
                colors = frame[edge_y, edge_x][:, ::-1] / 255.0

                # 1. Accumulate into Open3D 3D Point Cloud
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
                map_pcd = map_pcd.voxel_down_sample(voxel_size=0.05)

                # 2. Project points onto the 2D Top-Down Floor Plan (X and Z coordinates)
                # Ignore Y (height), scale and draw on grid map
                gx = np.int32(pts_world[:, 0] * draw_scale) + center_grid
                # Negate Z component to map forward motion upwards on the 2D canvas
                gy = np.int32(-pts_world[:, 2] * draw_scale) + center_grid
                
                # Draw boundary pixels inside the 2D canvas boundaries
                valid_mask = (gx >= 0) & (gx < grid_map_size) & (gy >= 0) & (gy < grid_map_size)
                grid_map[gy[valid_mask], gx[valid_mask]] = 255

        # --- STEP 5: Update Open3D Renderer ---
        pts_arr = np.array(path_points)
        num_pts = len(pts_arr)
        
        # Update path LineSet geometry
        if num_pts > 1:
            lines = [[i, i + 1] for i in range(num_pts - 1)]
            path_colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]
            
            path_line_set.points = o3d.utility.Vector3dVector(pts_arr)
            path_line_set.lines = o3d.utility.Vector2iVector(lines)
            path_line_set.colors = o3d.utility.Vector3dVector(path_colors)
            
            vis.remove_geometry(path_line_set, reset_bounding_box=False)
            vis.add_geometry(path_line_set, reset_bounding_box=False)

        # Refresh map PointCloud
        vis.remove_geometry(map_pcd, reset_bounding_box=False)
        vis.add_geometry(map_pcd, reset_bounding_box=False)

        # Trigger viewport auto-focus once at beginning
        if frame_count == 15:
            vis.reset_view_point(True)

        vis.poll_events()
        vis.update_renderer()

        # --- STEP 6: Render 2D SLAM Dashboard (BGR + Gray Edges + 2D Floor Plan) ---
        # 1. Left Color Feed (resized to 320x240)
        color_small = cv2.resize(frame, (320, 240))
        # Draw tracked green keypoints
        if vo.prev_pts is not None:
            # Scale coordinates down to match 320x240 resolution
            scale_x = 320.0 / w
            scale_y = 240.0 / h
            for pt in vo.prev_pts:
                cv2.circle(color_small, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 2, (0, 255, 0), -1)
        
        # Overlay coordinates
        pos_text = f"X:{tx:.2f} Y:{ty:.2f} Z:{tz:.2f}"
        cv2.putText(color_small, pos_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(color_small, "COLOR FEED", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Left Grayscale Edges Feed (resized to 320x240)
        edges_small = cv2.resize(edges, (320, 240))
        edges_small_bgr = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)
        cv2.putText(edges_small_bgr, "GRAYSCALE EDGES", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Stack left column vertically (Height: 480, Width: 320)
        left_col = np.vstack((color_small, edges_small_bgr))

        # 3. Right 2D Floor Plan Grid Overlay
        grid_display = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)
        # Draw historical 2D trajectory path (red lines)
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            g1x = int(p1[0] * draw_scale) + center_grid
            g1y = int(-p1[2] * draw_scale) + center_grid
            g2x = int(p2[0] * draw_scale) + center_grid
            g2y = int(-p2[2] * draw_scale) + center_grid
            if (0 <= g1x < grid_map_size and 0 <= g1y < grid_map_size and 
                0 <= g2x < grid_map_size and 0 <= g2y < grid_map_size):
                cv2.line(grid_display, (g1x, g1y), (g2x, g2y), (0, 0, 255), 1)
        
        # Draw current camera position (green dot)
        cx_grid = int(tx * draw_scale) + center_grid
        cy_grid = int(-tz * draw_scale) + center_grid
        if 0 <= cx_grid < grid_map_size and 0 <= cy_grid < grid_map_size:
            cv2.circle(grid_display, (cx_grid, cy_grid), 4, (0, 255, 0), -1)

        cv2.putText(grid_display, "2D TOP-DOWN MAP", (10, grid_map_size - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Horizontal stack Left Column and Right 2D Floor Plan (Height: 480, Width: 800)
        dashboard = np.hstack((left_col, grid_display))
        cv2.imshow("SLAM Dashboard (CV + 2D Floor Plan)", dashboard)

        # Listen for quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[SLAM] Trajectory tracking stopped.")

    # Save the Point Cloud Map on Exit
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
