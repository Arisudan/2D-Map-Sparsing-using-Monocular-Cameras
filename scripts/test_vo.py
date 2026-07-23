import cv2
import numpy as np
import sys
import os
import open3d as o3d
from visual_odometry import VisualOdometry

def main():
    # 1. Initialize Visual Odometry
    vo = VisualOdometry(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    # 2. Setup video source (Webcam, MP4 file, or stream URL)
    video_source = 0  # Default to default webcam (device index 0)
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        if source_arg.isdigit():
            video_source = int(source_arg)
        elif os.path.exists(source_arg) or source_arg.startswith("rtsp://") or source_arg.startswith("rtmp://"):
            video_source = source_arg
            print(f"[VO Test] Loading video stream: {video_source}")
        else:
            print(f"[VO Test] Error: Video file or stream '{source_arg}' not found.")
            return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[VO Test] Error: Could not open video source '{video_source}'")
        return

    print("[VO Test] Video source opened successfully.")
    print("[VO Test] Press 'q' or 'ESC' inside the video window to quit.")

    # Store 3D camera trajectory points
    path_points = [np.array([0.0, 0.0, 0.0])]

    # 3. Initialize Open3D Visualizer for 3D trajectory visualization on the right
    print("[VO Test] Initializing Open3D Visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Camera Trajectory (Open3D)", width=600, height=600, left=650, top=50)
    
    # Create LineSet to draw camera path in 3D and initialize it with the origin point
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(path_points))
    vis.add_geometry(line_set)

    # Add a coordinate axis helper at the origin (Red = X, Green = Y, Blue = Z) for visual reference
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis_frame)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[VO Test] Video feed complete or interrupted.")
            break

        # 4. Run visual odometry tracking
        R, t = vo.process_frame(frame)
        x, y, z = t[0, 0], t[1, 0], t[2, 0]
        
        # Append the new 3D position to our trajectory path
        # In ROS coordinate frame: X is right, Y is down, Z is forward
        # Open3D coordinate frame: X is right, Y is up, Z is backward (we negate Y/Z for natural view)
        path_points.append(np.array([x, -y, -z]))

        # 5. Prepare the Left Color Frame (with overlayed green keypoints)
        color_frame = frame.copy()
        if vo.prev_pts is not None:
            for pt in vo.prev_pts:
                cv2.circle(color_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # Print positional information onto the screen overlay
        pos_text = f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
        cv2.putText(color_frame, pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(color_frame, f"Tracked Pts: {len(vo.prev_pts) if vo.prev_pts is not None else 0}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_frame, "LIVE COLOR FEED", (20, color_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. Prepare the Center Black-and-White Frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR so we can stack it horizontally with color_frame
        gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        if vo.prev_pts is not None:
            for pt in vo.prev_pts:
                cv2.circle(gray_frame_bgr, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1)
        cv2.putText(gray_frame_bgr, "GRAYSCALE TRACKING", (20, gray_frame_bgr.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 7. Stack Color and Grayscale horizontally into a single dashboard window
        dashboard = np.hstack((color_frame, gray_frame_bgr))

        # 8. Update Open3D 3D Trajectory geometry
        pts_arr = np.array(path_points)
        num_pts = len(pts_arr)
        
        if num_pts > 1:
            lines = [[i, i + 1] for i in range(num_pts - 1)]
            colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]  # Red lines
            
            line_set.points = o3d.utility.Vector3dVector(pts_arr)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            # Remove and re-add geometry to force Open3D to reallocate OpenGL buffers for new points/lines
            vis.remove_geometry(line_set, reset_bounding_box=False)
            vis.add_geometry(line_set, reset_bounding_box=False)
            
            # Reset viewport once early on (at 10 points) to auto-focus on the path
            if num_pts == 10:
                vis.reset_view_point(True)
        
        # Poll events to update Open3D window dynamically without blocking
        vis.poll_events()
        vis.update_renderer()

        # 9. Render the OpenCV split dashboard
        cv2.imshow("SLAM Dashboard (Color | Grayscale)", dashboard)

        # Listen for quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[VO Test] Trajectory tracking stopped.")

if __name__ == "__main__":
    main()
