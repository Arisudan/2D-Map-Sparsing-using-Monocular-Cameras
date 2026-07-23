import socket
import struct
import cv2
import numpy as np
import sys
import os
import open3d as o3d
import datetime
from visual_odometry import VisualOdometry

def recv_all(sock, count):
    """Utility to receive exactly count bytes from a socket."""
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def icp_2d(ref_pts, src_pts, max_iterations=15, tolerance=1e-4):
    """
    2D Iterative Closest Point (ICP) scan-matching.
    Aligns src_pts (M, 2) to ref_pts (N, 2) using Open3D's C++ KDTree.
    Returns 2D Rotation matrix (2x2) and 2D Translation vector (2x1).
    """
    if len(ref_pts) < 20 or len(src_pts) < 20:
        return np.eye(2), np.zeros((2, 1))

    # Convert 2D points to 3D for Open3D's fast KDTree (setting Y = 0)
    ref_3d = np.zeros((len(ref_pts), 3))
    ref_3d[:, 0] = ref_pts[:, 0]
    ref_3d[:, 2] = ref_pts[:, 1]
    
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_3d)
    kdtree = o3d.geometry.KDTreeFlann(ref_pcd)
    
    R_accum = np.eye(2)
    t_accum = np.zeros((2, 1))
    
    src_aligned = src_pts.copy()
    prev_error = 9999.0
    
    for step in range(max_iterations):
        closest_pts = []
        valid_src = []
        errors = []
        
        for pt in src_aligned:
            # Query 1 nearest neighbor
            query_pt = np.array([pt[0], 0.0, pt[1]])
            [_, idx, dist_sq] = kdtree.search_knn_vector_3d(query_pt, 1)
            if len(idx) > 0:
                # Reject correspondences that are too far (> 40 cm) to avoid false matches
                if dist_sq[0] < 0.16:
                    closest_pts.append(ref_pts[idx[0]])
                    valid_src.append(pt)
                    errors.append(np.sqrt(dist_sq[0]))
        
        if len(valid_src) < 15:
            break
            
        valid_src = np.array(valid_src)
        closest_pts = np.array(closest_pts)
        
        # Compute centroids
        c_src = np.mean(valid_src, axis=0)
        c_ref = np.mean(closest_pts, axis=0)
        
        # Center points
        src_centered = valid_src - c_src
        ref_centered = closest_pts - c_ref
        
        # Compute covariance matrix H
        H = src_centered.T.dot(ref_centered)
        U, S, Vt = np.linalg.svd(H)
        R_step = Vt.T.dot(U.T)
        
        # Prevent reflection
        if np.linalg.det(R_step) < 0:
            Vt[1, :] *= -1
            R_step = Vt.T.dot(U.T)
            
        t_step = c_ref.reshape(2, 1) - R_step.dot(c_src.reshape(2, 1))
        
        # Accumulate transformation
        R_accum = R_step.dot(R_accum)
        t_accum = R_step.dot(t_accum) + t_step
        
        # Apply transformation to src_aligned for next iteration
        src_aligned = src_pts.dot(R_accum.T) + t_accum.T
        
        # Check error convergence
        mean_error = np.mean(errors)
        if abs(mean_error - prev_error) < tolerance:
            break
        prev_error = mean_error
        
    return R_accum, t_accum

def main():
    # Camera Intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0

    # 1. Initialize Visual Odometry locally (runs very fast on CPU)
    print("[SLAM Client] Initializing Visual Odometry...")
    vo = VisualOdometry(fx=fx, fy=fy, cx=cx, cy=cy)

    # 2. Connect to the RTX 5090 depth server via SSH Port Forwarding Tunnel
    print("[SLAM Client] Connecting to Depth Server on localhost:5000...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', 5000))
        print("[SLAM Client] Connection successful! Linked to RTX 5090 GPU.")
    except Exception as e:
        print(f"[SLAM Client] Connection failed: {e}")
        print("[SLAM Client] Please make sure:")
        print("  1. The server script is running on the RTX 5090 PC.")
        print("  2. Your SSH tunnel is established: ssh -L 5000:localhost:5000 user@rtx5090_ip")
        return

    # 3. Setup video source (Webcam, MP4 file, or stream URL)
    video_source = 0  # Default to default webcam (device index 0)
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        if source_arg.isdigit():
            video_source = int(source_arg)
        elif os.path.exists(source_arg) or source_arg.startswith("rtsp://") or source_arg.startswith("rtmp://"):
            video_source = source_arg
            print(f"[SLAM Client] Loading video source: {video_source}")
        else:
            print(f"[SLAM Client] Error: Video source '{source_arg}' not found.")
            client_socket.close()
            return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[SLAM Client] Error: Could not open video source '{video_source}'")
        client_socket.close()
        return

    print("[SLAM Client] Video source opened successfully.")
    print("[SLAM Client] Press 'q' or 'ESC' inside the dashboard window to quit.")

    # 4. Initialize Open3D Visualizer for 3D Sparse Edge Mapping on the right
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

    # 5. Initialize 2D Floor Plan Grid Map
    grid_map_size = 480
    grid_map = np.zeros((grid_map_size, grid_map_size), dtype=np.uint8)
    draw_scale = 50.0
    center_grid = grid_map_size // 2

    # Optimization parameters
    frame_count = 0
    depth_interval = 10  # Only request depth estimation once every 10 frames to save network bandwidth
    latest_depth_map = None
    latest_edges = None

    # Trajectory Memory (Low-pass filter) variables
    smooth_t = None
    alpha = 0.3  # Smoothing factor (lower = smoother, higher = more responsive)

    # 2D Map point registry for scan-matching / alignment (Step 3)
    map_pts_2d = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[SLAM Client] Video source finished or disconnected.")
                break

            frame_count += 1
            h, w = frame.shape[:2]
            cx, cy = w / 2.0, h / 2.0

            # --- STEP 1: Run Visual Odometry Tracking (Full Speed) ---
            R, t = vo.process_frame(frame)
            
            # Apply Trajectory Memory low-pass smoothing filter (Step 3)
            if smooth_t is None:
                smooth_t = t.copy()
            else:
                smooth_t = alpha * t + (1.0 - alpha) * smooth_t

            tx, ty, tz = smooth_t[0, 0], smooth_t[1, 0], smooth_t[2, 0]
            
            # Append to camera path using the smoothed coordinates
            path_points.append(np.array([tx, -ty, -tz]))

            # --- STEP 2: Network-Based Depth Request (Keyframe Filter) ---
            is_keyframe = (frame_count % depth_interval == 1) or (latest_depth_map is None)
            
            if is_keyframe:
                # 1. Compress current frame to JPEG (JPEG quality: 85)
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = img_encoded.tobytes()

                # 2. Send the image size + raw image bytes to the server
                client_socket.sendall(struct.pack('>I', len(img_bytes)) + img_bytes)

                # 3. Read the size of the returned 16-bit PNG depth map
                len_bytes = recv_all(client_socket, 4)
                if len_bytes is None:
                    print("[SLAM Client] Server connection lost.")
                    break
                depth_len = struct.unpack('>I', len_bytes)[0]

                # 4. Receive the raw depth PNG bytes
                depth_png_bytes = recv_all(client_socket, depth_len)
                if depth_png_bytes is None:
                    print("[SLAM Client] Server connection lost.")
                    break

                # 5. Decode PNG depth map and convert back to relative scale
                depth_arr = np.frombuffer(depth_png_bytes, dtype=np.uint8)
                depth_uint16 = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
                if depth_uint16 is None:
                    print("[SLAM Client] Error: Failed to decode depth map PNG from server.")
                    continue
                
                raw_depth = depth_uint16.astype(np.float32) / 1000.0

                # Normalize metric scale assuming an indoor median-depth of 2.0 meters
                median_depth = np.median(raw_depth)
                scaled_depth = raw_depth * (2.0 / (median_depth + 1e-6))
                latest_depth_map = np.clip(scaled_depth, 0.1, 10.0)

                # --- STEP 3: Detect Depth-Gradient Edges (AI Physical Boundaries) ---
                # We normalize the depth map to 0-255 uint8 and run Canny on it.
                # This ignores shadows/colors and captures pure physical depth changes.
                depth_norm = cv2.normalize(latest_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                latest_edges = cv2.Canny(depth_norm, 15, 45)

            # Ensure we have a valid edges frame for display
            edges = latest_edges if latest_edges is not None else np.zeros((h, w), dtype=np.uint8)

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
                    
                    pts_camera = np.stack((X, -Y, -Z), axis=-1)
                    
                    # Transform camera points to world frame
                    t_world = np.array([tx, -ty, -tz])
                    pts_world = pts_camera.dot(R.T) + t_world
                    
                    # Apply 2D ICP Aligner to snap boundaries and correct camera drift
                    src_pts_2d = pts_world[:, [0, 2]]
                    
                    if map_pts_2d is not None:
                        # Calculate alignment correction
                        R_2d, t_2d = icp_2d(map_pts_2d, src_pts_2d)
                        
                        # Correct projected 3D points
                        pts_world[:, [0, 2]] = src_pts_2d.dot(R_2d.T) + t_2d.T
                        
                        # Correct camera trajectory coordinates
                        cam_pos_2d = np.array([tx, -tz]).reshape(2, 1)
                        cam_pos_corrected = R_2d.dot(cam_pos_2d) + t_2d
                        tx = cam_pos_corrected[0, 0]
                        tz = -cam_pos_corrected[1, 0]
                        
                        # Update the latest coordinate in the path history
                        path_points[-1] = np.array([tx, -ty, -tz])
                    
                    # Accumulate corrected points into registry
                    if map_pts_2d is None:
                        map_pts_2d = pts_world[:, [0, 2]]
                    else:
                        map_pts_2d = np.vstack((map_pts_2d, pts_world[:, [0, 2]]))
                        # Keep registry capped at 15000 points to keep search fast
                        if len(map_pts_2d) > 15000:
                            map_pts_2d = map_pts_2d[-15000:]
                    
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

                    # 2. Project points onto the 2D Top-Down Floor Plan
                    gx = np.int32(pts_world[:, 0] * draw_scale) + center_grid
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

            if frame_count == 15:
                vis.reset_view_point(True)

            vis.poll_events()
            vis.update_renderer()

            # --- STEP 6: Render 2D SLAM Dashboard ---
            # 1. Left Color Feed (resized to 320x240)
            color_small = cv2.resize(frame, (320, 240))
            if vo.prev_pts is not None:
                scale_x = 320.0 / w
                scale_y = 240.0 / h
                for pt in vo.prev_pts:
                    cv2.circle(color_small, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 2, (0, 255, 0), -1)
            
            pos_text = f"X:{tx:.2f} Y:{ty:.2f} Z:{tz:.2f}"
            cv2.putText(color_small, pos_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(color_small, "COLOR FEED (30 FPS)", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 2. Left Grayscale Edges Feed (resized to 320x240)
            edges_small = cv2.resize(edges, (320, 240))
            edges_small_bgr = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)
            cv2.putText(edges_small_bgr, "GRAYSCALE EDGES", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            left_col = np.vstack((color_small, edges_small_bgr))

            # 3. Right 2D Floor Plan Grid Overlay
            grid_display = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)
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
            
            cx_grid = int(tx * draw_scale) + center_grid
            cy_grid = int(-tz * draw_scale) + center_grid
            if 0 <= cx_grid < grid_map_size and 0 <= cy_grid < grid_map_size:
                cv2.circle(grid_display, (cx_grid, cy_grid), 4, (0, 255, 0), -1)

            cv2.putText(grid_display, "2D TOP-DOWN MAP", (10, grid_map_size - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            dashboard = np.hstack((left_col, grid_display))
            cv2.imshow("SLAM Client Dashboard (CV + 2D Floor Plan)", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[SLAM Client] Stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        client_socket.close()
        print("[SLAM Client] Disconnected from server.")

    # Save the Point Cloud Map on Exit
    if len(map_pcd.points) > 0:
        os.makedirs("maps", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maps/Map1_{timestamp}.ply"
        print(f"[SLAM Client] Saving accumulated 3D map to '{filename}'...")
        success = o3d.io.write_point_cloud(filename, map_pcd)
        if success:
            print("[SLAM Client] Map saved successfully.")
        else:
            print("[SLAM Client] Error: Failed to save map.")

if __name__ == "__main__":
    main()
