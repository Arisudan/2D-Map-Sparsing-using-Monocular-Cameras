import os
# Force Hugging Face and Torch to download and cache models on the D drive
os.environ["HF_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/huggingface"
os.environ["TORCH_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/torch"

import socket
import struct
import cv2
import numpy as np
import sys
import open3d as o3d
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from visual_odometry import VisualOdometry

# --- 2D ICP Scan-Matching Aligner ---
def icp_2d(ref_pts, src_pts, max_iterations=15, tolerance=1e-4):
    """
    2D Iterative Closest Point (ICP) scan-matching.
    Aligns src_pts (M, 2) to ref_pts (N, 2) using Open3D's C++ KDTree.
    """
    if len(ref_pts) < 20 or len(src_pts) < 20:
        return np.eye(2), np.zeros((2, 1))

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
            query_pt = np.array([pt[0], 0.0, pt[1]])
            [_, idx, dist_sq] = kdtree.search_knn_vector_3d(query_pt, 1)
            if len(idx) > 0:
                if dist_sq[0] < 0.16:  # Reject points further than 40cm
                    closest_pts.append(ref_pts[idx[0]])
                    valid_src.append(pt)
                    errors.append(np.sqrt(dist_sq[0]))
        
        if len(valid_src) < 15:
            break
            
        valid_src = np.array(valid_src)
        closest_pts = np.array(closest_pts)
        
        c_src = np.mean(valid_src, axis=0)
        c_ref = np.mean(closest_pts, axis=0)
        
        src_centered = valid_src - c_src
        ref_centered = closest_pts - c_ref
        
        H = src_centered.T.dot(ref_centered)
        U, S, Vt = np.linalg.svd(H)
        R_step = Vt.T.dot(U.T)
        
        if np.linalg.det(R_step) < 0:
            Vt[1, :] *= -1
            R_step = Vt.T.dot(U.T)
            
        t_step = c_ref.reshape(2, 1) - R_step.dot(c_src.reshape(2, 1))
        
        R_accum = R_step.dot(R_accum)
        t_accum = R_step.dot(t_accum) + t_step
        
        src_aligned = src_pts.dot(R_accum.T) + t_accum.T
        
        mean_error = np.mean(errors)
        if abs(mean_error - prev_error) < tolerance:
            break
        prev_error = mean_error
        
    return R_accum, t_accum

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


class SLAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Monocular SLAM Dashboard")
        self.root.geometry("900x700")
        self.root.configure(bg="#121212")

        # Camera Intrinsics
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0

        # State Variables
        self.running = False
        self.selected_source = "0"  # Default to Webcam (Camera 0)
        self.video_file_path = ""
        self.show_edges = False     # Toggle between Color and Edges in PiP
        self.pip_expanded = False   # PiP window size state

        # 2D Map Canvas Setup
        self.map_size = 700
        self.grid_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.draw_scale = 65.0
        self.center_grid = self.map_size // 2

        # 3D Mapping Geometry (retained for saving PLY)
        self.map_pcd = o3d.geometry.PointCloud()

        # Build GUI Layout
        self.build_gui()

    def build_gui(self):
        # 1. Main Background Map Label (Fills entire screen)
        self.map_label = tk.Label(self.root, bg="#121212")
        self.map_label.pack(fill=tk.BOTH, expand=True)

        # Draw initial blank map
        self.update_map_display()

        # 2. FLOATING CONTROL PANEL (Top-Left)
        self.ctrl_frame = tk.Frame(self.root, bg="#1e1e1e", bd=2, relief=tk.SOLID, padx=10, pady=10)
        self.ctrl_frame.place(x=15, y=15, anchor="nw")

        # Header
        lbl_title = tk.Label(self.ctrl_frame, text="SLAM CONTROLLER", fg="#00ff00", bg="#1e1e1e", font=("Arial", 10, "bold"))
        lbl_title.pack(anchor="w", pady=(0, 10))

        # Source Selection Radio Buttons
        self.source_var = tk.StringVar(value="0")
        
        rb_cam0 = tk.Radiobutton(self.ctrl_frame, text="Camera 0 (Webcam)", variable=self.source_var, value="0", 
                                 fg="#ffffff", bg="#1e1e1e", selectcolor="#1e1e1e", activebackground="#1e1e1e", command=self.on_source_change)
        rb_cam0.pack(anchor="w")

        rb_cam1 = tk.Radiobutton(self.ctrl_frame, text="Camera 1 (Drone Card)", variable=self.source_var, value="1", 
                                 fg="#ffffff", bg="#1e1e1e", selectcolor="#1e1e1e", activebackground="#1e1e1e", command=self.on_source_change)
        rb_cam1.pack(anchor="w")

        rb_cam2 = tk.Radiobutton(self.ctrl_frame, text="Camera 2 (Auxiliary)", variable=self.source_var, value="2", 
                                 fg="#ffffff", bg="#1e1e1e", selectcolor="#1e1e1e", activebackground="#1e1e1e", command=self.on_source_change)
        rb_cam2.pack(anchor="w")

        rb_file = tk.Radiobutton(self.ctrl_frame, text="Local Video File", variable=self.source_var, value="file", 
                                 fg="#ffffff", bg="#1e1e1e", selectcolor="#1e1e1e", activebackground="#1e1e1e", command=self.on_source_change)
        rb_file.pack(anchor="w", pady=(5, 0))

        # File Browser Button
        self.btn_browse = tk.Button(self.ctrl_frame, text="Browse Video...", command=self.browse_file, state=tk.DISABLED,
                                    bg="#333333", fg="#ffffff", activebackground="#555555", activeforeground="#ffffff", bd=0, padx=10, pady=2)
        self.btn_browse.pack(fill=tk.X, pady=(5, 10))

        self.lbl_filename = tk.Label(self.ctrl_frame, text="No file selected", fg="#aaaaaa", bg="#1e1e1e", wraplength=180, justify=tk.LEFT)
        self.lbl_filename.pack(anchor="w", pady=(0, 10))

        # Operation Buttons
        self.btn_start = tk.Button(self.ctrl_frame, text="START SLAM", command=self.start_slam, bg="#006600", fg="#ffffff",
                                   font=("Arial", 9, "bold"), activebackground="#009900", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_start.pack(fill=tk.X, pady=(0, 5))

        self.btn_stop = tk.Button(self.ctrl_frame, text="STOP & SAVE MAP", command=self.stop_slam, state=tk.DISABLED, bg="#990000", fg="#ffffff",
                                  font=("Arial", 9, "bold"), activebackground="#cc0000", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_stop.pack(fill=tk.X)

        # 3. FLOATING VIDEO PIP PANEL (Top-Right)
        self.pip_frame = tk.Frame(self.root, bg="#1e1e1e", bd=2, relief=tk.SOLID)
        self.pip_frame.place(relx=1.0, rely=0.0, x=-15, y=15, anchor="ne")

        # PiP Header Bar
        self.pip_header = tk.Frame(self.pip_frame, bg="#2d2d2d", height=24)
        self.pip_header.pack(fill=tk.X)
        self.pip_header.pack_propagate(False)

        self.btn_toggle_view = tk.Button(self.pip_header, text="View: Color", command=self.toggle_pip_view, bg="#444444", fg="#ffffff",
                                         font=("Arial", 7, "bold"), activebackground="#666666", activeforeground="#ffffff", bd=0, padx=5)
        self.btn_toggle_view.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        self.btn_resize = tk.Button(self.pip_header, text="[-]", command=self.toggle_pip_size, bg="#444444", fg="#ffffff",
                                    font=("Arial", 7, "bold"), activebackground="#666666", activeforeground="#ffffff", bd=0, padx=5)
        self.btn_resize.pack(side=tk.RIGHT, fill=tk.Y, padx=2, pady=2)

        # PiP Image Display
        self.pip_width = 160
        self.pip_height = 120
        self.pip_label = tk.Label(self.pip_frame, bg="#000000", width=self.pip_width, height=self.pip_height)
        self.pip_label.pack()

        # Bind image click to expand/shrink
        self.pip_label.bind("<Button-1>", lambda e: self.toggle_pip_size())

    def on_source_change(self):
        self.selected_source = self.source_var.get()
        if self.selected_source == "file":
            self.btn_browse.config(state=tk.NORMAL)
        else:
            self.btn_browse.config(state=tk.DISABLED)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_file_path = file_path
            filename = os.path.basename(file_path)
            self.lbl_filename.config(text=filename, fg="#00ff00")

    def toggle_pip_view(self):
        self.show_edges = not self.show_edges
        self.btn_toggle_view.config(text="View: Edges" if self.show_edges else "View: Color")

    def toggle_pip_size(self):
        self.pip_expanded = not self.pip_expanded
        if self.pip_expanded:
            self.pip_width = 320
            self.pip_height = 240
            self.btn_resize.config(text="[+]")
        else:
            self.pip_width = 160
            self.pip_height = 120
            self.btn_resize.config(text="[-]")
        
        self.pip_label.config(width=self.pip_width, height=self.pip_height)

    def update_map_display(self, map_image=None):
        """Draws the current 2D sparse map array onto the background label."""
        if map_image is None:
            # Draw standard empty map canvas with coordinate axis
            map_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            cv2.line(map_image, (self.center_grid, 0), (self.center_grid, self.map_size), (40, 40, 40), 1)
            cv2.line(map_image, (0, self.center_grid), (self.map_size, self.center_grid), (40, 40, 40), 1)
        
        # Convert BGR to RGB for PIL
        rgb_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.tk_map_img = ImageTk.PhotoImage(pil_img)
        self.map_label.config(image=self.tk_map_img)

    def start_slam(self):
        if self.running:
            return

        # Determine source argument
        if self.selected_source == "file":
            if not self.video_file_path:
                messagebox.showerror("Error", "Please select a local video file first.")
                return
            self.source_arg = self.video_file_path
        else:
            self.source_arg = int(self.selected_source)

        # Reset map variables
        self.grid_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.map_pcd = o3d.geometry.PointCloud()

        # Update button states
        self.running = True
        self.btn_start.config(state=tk.DISABLED, bg="#333333")
        self.btn_stop.config(state=tk.NORMAL)
        
        # Disable source selections while SLAM runs
        self.btn_browse.config(state=tk.DISABLED)

        # Launch the client thread
        self.client_thread = threading.Thread(target=self.slam_worker_thread, daemon=True)
        self.client_thread.start()

    def stop_slam(self):
        if not self.running:
            return
        
        self.running = False
        self.btn_stop.config(state=tk.DISABLED)

    def slam_worker_thread(self):
        """Background thread handling video capture, visual odometry, and server requests."""
        # Connect to RTX 5090 depth server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect(('localhost', 5000))
        except Exception as e:
            messagebox.showerror("Connection Error", 
                                 f"Failed to connect to GPU Server on localhost:5000.\n{e}\n\n"
                                 "Make sure your SSH tunnel is open and the server script is running.")
            self.root.after(0, self.reset_gui_state)
            return

        cap = cv2.VideoCapture(self.source_arg)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open video source: {self.source_arg}")
            client_socket.close()
            self.root.after(0, self.reset_gui_state)
            return

        # Initialize Visual Odometry
        vo = VisualOdometry(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

        # Trajectory & SLAM Alignment States
        path_points = [np.array([0.0, 0.0, 0.0])]
        smooth_t = None
        alpha = 0.3
        map_pts_2d = None

        frame_count = 0
        depth_interval = 10
        latest_depth_map = None
        latest_edges = None

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[SLAM Client] Video feed completed.")
                break

            frame_count += 1
            h, w = frame.shape[:2]
            cx, cy = w / 2.0, h / 2.0

            # --- STEP 1: Local Visual Odometry ---
            R, t = vo.process_frame(frame)
            
            # Apply low-pass Jitter Filter
            if smooth_t is None:
                smooth_t = t.copy()
            else:
                smooth_t = alpha * t + (1.0 - alpha) * smooth_t

            tx, ty, tz = smooth_t[0, 0], smooth_t[1, 0], smooth_t[2, 0]
            path_points.append(np.array([tx, -ty, -tz]))

            # --- STEP 2: Offload Depth Request (Keyframe Filter) ---
            is_keyframe = (frame_count % depth_interval == 1) or (latest_depth_map is None)
            
            if is_keyframe:
                # Compress to JPEG
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = img_encoded.tobytes()

                try:
                    # Send size + frame
                    client_socket.sendall(struct.pack('>I', len(img_bytes)) + img_bytes)
                    # Receive size + PNG depth map
                    len_bytes = recv_all(client_socket, 4)
                    if len_bytes is None:
                        break
                    depth_len = struct.unpack('>I', len_bytes)[0]
                    depth_png_bytes = recv_all(client_socket, depth_len)
                    if depth_png_bytes is None:
                        break
                except Exception as e:
                    print(f"[SLAM Client] Network streaming error: {e}")
                    break

                # Decode PNG Depth
                depth_arr = np.frombuffer(depth_png_bytes, dtype=np.uint8)
                depth_uint16 = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
                if depth_uint16 is not None:
                    raw_depth = depth_uint16.astype(np.float32) / 1000.0
                    median_depth = np.median(raw_depth)
                    scaled_depth = raw_depth * (2.0 / (median_depth + 1e-6))
                    latest_depth_map = np.clip(scaled_depth, 0.1, 10.0)

                    # --- STEP 3: Depth-Gradient Edge Boundaries ---
                    depth_norm = cv2.normalize(latest_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    latest_edges = cv2.Canny(depth_norm, 15, 45)

            # Retrieve active edge frame
            edges = latest_edges if latest_edges is not None else np.zeros((h, w), dtype=np.uint8)

            # --- STEP 4: 2D ICP Alignment & Mapping ---
            if is_keyframe and latest_depth_map is not None:
                edge_y, edge_x = np.where(edges > 0)
                
                # Subsample to avoid CPU lag
                if len(edge_x) > 1000:
                    indices = np.random.choice(len(edge_x), size=1000, replace=False)
                    edge_x = edge_x[indices]
                    edge_y = edge_y[indices]

                if len(edge_x) > 0:
                    Z = latest_depth_map[edge_y, edge_x]
                    X = (edge_x - cx) * Z / self.fx
                    Y = (edge_y - cy) * Z / self.fy
                    
                    pts_camera = np.stack((X, -Y, -Z), axis=-1)
                    t_world = np.array([tx, -ty, -tz])
                    pts_world = pts_camera.dot(R.T) + t_world
                    
                    # Run 2D ICP Alignment (Map-to-Scan)
                    src_pts_2d = pts_world[:, [0, 2]]
                    
                    if map_pts_2d is not None:
                        R_2d, t_2d = icp_2d(map_pts_2d, src_pts_2d)
                        
                        # Correct 3D points
                        pts_world[:, [0, 2]] = src_pts_2d.dot(R_2d.T) + t_2d.T
                        
                        # Correct camera trajectory coordinates
                        cam_pos_2d = np.array([tx, -tz]).reshape(2, 1)
                        cam_pos_corrected = R_2d.dot(cam_pos_2d) + t_2d
                        tx = cam_pos_corrected[0, 0]
                        tz = -cam_pos_corrected[1, 0]
                        
                        # Correct latest coordinate in path
                        path_points[-1] = np.array([tx, -ty, -tz])
                    
                    # Accumulate corrected points
                    if map_pts_2d is None:
                        map_pts_2d = pts_world[:, [0, 2]]
                    else:
                        map_pts_2d = np.vstack((map_pts_2d, pts_world[:, [0, 2]]))
                        if len(map_pts_2d) > 15000:
                            map_pts_2d = map_pts_2d[-15000:]

                    # Accumulate into 3D Point Cloud geometry (for saving PLY file on exit)
                    colors = frame[edge_y, edge_x][:, ::-1] / 255.0
                    old_pts = np.asarray(self.map_pcd.points)
                    old_cols = np.asarray(self.map_pcd.colors)
                    if len(old_pts) > 0:
                        merged_pts = np.vstack((old_pts, pts_world))
                        merged_cols = np.vstack((old_cols, colors))
                    else:
                        merged_pts = pts_world
                        merged_cols = colors
                    self.map_pcd.points = o3d.utility.Vector3dVector(merged_pts)
                    self.map_pcd.colors = o3d.utility.Vector3dVector(merged_cols)
                    self.map_pcd = self.map_pcd.voxel_down_sample(voxel_size=0.05)

                    # Project points onto 2D Top-Down Floor Plan
                    gx = np.int32(pts_world[:, 0] * self.draw_scale) + self.center_grid
                    gy = np.int32(-pts_world[:, 2] * self.draw_scale) + self.center_grid
                    
                    valid_mask = (gx >= 0) & (gx < self.map_size) & (gy >= 0) & (gy < self.map_size)
                    self.grid_map[gy[valid_mask], gx[valid_mask]] = 255

            # --- STEP 5: Update GUI Visual Elements (Thread-Safe) ---
            # 1. Prepare Background Map Image
            map_image = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
            # Draw axis lines
            cv2.line(map_image, (self.center_grid, 0), (self.center_grid, self.map_size), (40, 40, 40), 1)
            cv2.line(map_image, (0, self.center_grid), (self.map_size, self.center_grid), (40, 40, 40), 1)
            
            # Draw historical 2D trajectory path
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i + 1]
                g1x = int(p1[0] * self.draw_scale) + self.center_grid
                g1y = int(-p1[2] * self.draw_scale) + self.center_grid
                g2x = int(p2[0] * self.draw_scale) + self.center_grid
                g2y = int(-p2[2] * self.draw_scale) + self.center_grid
                if (0 <= g1x < self.map_size and 0 <= g1y < self.map_size and 
                    0 <= g2x < self.map_size and 0 <= g2y < self.map_size):
                    cv2.line(map_image, (g1x, g1y), (g2x, g2y), (0, 0, 255), 1)
            
            # Draw current camera position (green dot)
            cx_grid = int(tx * self.draw_scale) + self.center_grid
            cy_grid = int(-tz * self.draw_scale) + self.center_grid
            if 0 <= cx_grid < self.map_size and 0 <= cy_grid < self.map_size:
                cv2.circle(map_image, (cx_grid, cy_grid), 5, (0, 255, 0), -1)

            # 2. Prepare PiP Video Overlay
            # Toggle between color and edge outlines
            if self.show_edges:
                pip_frame = cv2.resize(edges, (self.pip_width, self.pip_height))
                pip_frame_rgb = cv2.cvtColor(pip_frame, cv2.COLOR_GRAY2RGB)
            else:
                # Resize color frame to PiP dimensions and draw tracked features
                color_pip = frame.copy()
                if vo.prev_pts is not None:
                    scale_x = float(self.pip_width) / w
                    scale_y = float(self.pip_height) / h
                    for pt in vo.prev_pts:
                        cv2.circle(color_pip, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 2, (0, 255, 0), -1)
                resized_color = cv2.resize(color_pip, (self.pip_width, self.pip_height))
                pip_frame_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)

            # Schedule GUI elements update on main Tkinter thread
            self.root.after(0, self.update_gui_frames, map_image, pip_frame_rgb)

        # Cleanup
        cap.release()
        client_socket.close()
        print("[SLAM Client] Worker thread stopped.")

        # Save map on exit
        self.root.after(0, self.save_map_and_reset)

    def update_gui_frames(self, map_image, pip_frame_rgb):
        """Thread-safe update of Tkinter widgets using PIL."""
        # 1. Update full-screen background map
        self.update_map_display(map_image)

        # 2. Update top-right floating PiP frame
        pil_pip = Image.fromarray(pip_frame_rgb)
        self.tk_pip_img = ImageTk.PhotoImage(pil_pip)
        self.pip_label.config(image=self.tk_pip_img)

    def save_map_and_reset(self):
        # Save point cloud PLY map
        if len(self.map_pcd.points) > 0:
            os.makedirs("maps", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maps/Map1_{timestamp}.ply"
            print(f"[SLAM Client] Saving accumulated 3D map to '{filename}'...")
            success = o3d.io.write_point_cloud(filename, self.map_pcd)
            if success:
                messagebox.showinfo("Success", f"SLAM Mapping complete!\n3D Map successfully saved to:\n{filename}")
            else:
                messagebox.showerror("Error", "Failed to save the 3D PLY map.")
        else:
            messagebox.showwarning("Warning", "SLAM stopped. Map is empty, nothing to save.")

        self.reset_gui_state()

    def reset_gui_state(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL, bg="#006600")
        self.btn_stop.config(state=tk.DISABLED)
        self.on_source_change()  # Re-evaluate Browse button state

        # Clear previews
        self.update_map_display()
        self.pip_label.config(image="")


if __name__ == "__main__":
    root = tk.Tk()
    app = SLAMApp(root)
    root.mainloop()
