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
        # Check error convergence
        mean_error = np.mean(errors)
        if abs(mean_error - prev_error) < tolerance:
            break
        prev_error = mean_error
        
    return R_accum, t_accum

def draw_camera_frustum_2d(img, tx, tz, R, scale, cx_draw, cy_draw, color, thickness=1):
    """
    Draws a 2D orthographic camera frustum (pyramid) pointing in the direction of R.
    """
    # Camera center in grid coordinates
    g_cx = int(tx * scale) + cx_draw
    g_cy = int(-tz * scale) + cy_draw

    # Left/right corners in camera coordinates (looks along -Z in our convention)
    d = 0.20
    w_half = 0.10
    p_left_cam = np.array([-w_half, 0, -d])
    p_right_cam = np.array([w_half, 0, -d])

    # Rotate to world coordinates
    p_left_world = p_left_cam.dot(R.T) + np.array([tx, 0.0, -tz])
    p_right_world = p_right_cam.dot(R.T) + np.array([tx, 0.0, -tz])

    # Project to 2D grid coordinates
    g_lx = int(p_left_world[0] * scale) + cx_draw
    g_ly = int(-p_left_world[2] * scale) + cy_draw
    
    g_rx = int(p_right_world[0] * scale) + cx_draw
    g_ry = int(-p_right_world[2] * scale) + cy_draw

    # Draw wireframe triangle lines
    cv2.line(img, (g_cx, g_cy), (g_lx, g_ly), color, thickness)
    cv2.line(img, (g_cx, g_cy), (g_rx, g_ry), color, thickness)
    cv2.line(img, (g_lx, g_ly), (g_rx, g_ry), color, thickness)

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

        # 2D Map Canvas Setup (Dynamic rendering parameters)
        self.map_size = 700
        self.draw_scale = 65.0
        self.center_grid = self.map_size // 2
        
        # SLAM Alignment & Map Registry States
        self.keyframe_poses = []   # List of tuples: (tx, ty, tz, R)
        self.all_map_points = None # Dynamic numpy array of shape (N, 2) storing world (X, Z)
        self.follow_camera = True   # Camera Follow active by default
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.last_tx = 0.0
        self.last_tz = 0.0
        self.last_R = np.eye(3)
        self.keyframe_data = []    # Detailed history of keyframes for Loop Closure

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

        # Bind mouse events for panning and zooming on the map
        self.map_label.bind("<ButtonPress-1>", self.start_pan)
        self.map_label.bind("<B1-Motion>", self.do_pan)
        self.map_label.bind("<MouseWheel>", self.zoom_map)
        self.map_label.bind("<Button-4>", self.zoom_map)  # Linux scroll up
        self.map_label.bind("<Button-5>", self.zoom_map)  # Linux scroll down

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

        # Follow Target Toggle Button (Default ON)
        self.btn_follow = tk.Button(self.ctrl_frame, text="Follow Target: ON", command=self.toggle_follow,
                                    bg="#006600", fg="#ffffff", font=("Arial", 9, "bold"), 
                                    activebackground="#009900", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_follow.pack(fill=tk.X, pady=(0, 10))

        # Operation Buttons
        self.btn_start = tk.Button(self.ctrl_frame, text="START SLAM", command=self.start_slam, bg="#006600", fg="#ffffff",
                                   font=("Arial", 9, "bold"), activebackground="#009900", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_start.pack(fill=tk.X, pady=(0, 5))

        self.btn_stop = tk.Button(self.ctrl_frame, text="STOP SLAM", command=self.stop_slam, state=tk.DISABLED, bg="#990000", fg="#ffffff",
                                  font=("Arial", 9, "bold"), activebackground="#cc0000", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_stop.pack(fill=tk.X, pady=(0, 5))

        self.btn_save = tk.Button(self.ctrl_frame, text="SAVE MAP", command=self.save_map, state=tk.DISABLED, bg="#333333", fg="#ffffff",
                                  font=("Arial", 9, "bold"), activebackground="#0055ff", activeforeground="#ffffff", bd=0, pady=5)
        self.btn_save.pack(fill=tk.X)

        # 3. FLOATING VIDEO PIP PANEL (Top-Right)
        self.pip_frame = tk.Frame(self.root, bg="#1e1e1e", bd=2, relief=tk.SOLID)
        self.pip_frame.place(relx=1.0, rely=0.0, x=-15, y=15, anchor="ne")

        # PiP Header Bar
        self.pip_header = tk.Frame(self.pip_frame, bg="#2d2d2d", height=24)
        self.pip_header.pack(fill=tk.X)
        self.pip_header.pack_propagate(False)

        # Bind drag events to the header frame
        self.pip_header.bind("<ButtonPress-1>", self.start_pip_drag)
        self.pip_header.bind("<B1-Motion>", self.do_pip_drag)

        # Title Label (packed left, acts as drag handle)
        self.lbl_pip_title = tk.Label(self.pip_header, text=" Feed (Drag)", fg="#aaaaaa", bg="#2d2d2d", font=("Arial", 7, "bold"))
        self.lbl_pip_title.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2))
        self.lbl_pip_title.bind("<ButtonPress-1>", self.start_pip_drag)
        self.lbl_pip_title.bind("<B1-Motion>", self.do_pip_drag)

        # Toggle view button
        self.btn_toggle_view = tk.Button(self.pip_header, text="Color", command=self.toggle_pip_view, bg="#444444", fg="#ffffff",
                                         font=("Arial", 7, "bold"), activebackground="#666666", activeforeground="#ffffff", bd=0, padx=5)
        self.btn_toggle_view.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        # Minimize/Maximize button
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
        self.btn_toggle_view.config(text="Edges" if self.show_edges else "Color")

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

    def start_pip_drag(self, event):
        """Record initial mouse click coordinates relative to the PiP frame."""
        self.pip_drag_start_x = event.x
        self.pip_drag_start_y = event.y

    def do_pip_drag(self, event):
        """Calculate and update absolute coordinate placement during dragging."""
        x = event.x_root - self.root.winfo_rootx() - self.pip_drag_start_x
        y = event.y_root - self.root.winfo_rooty() - self.pip_drag_start_y
        
        # Clamp to window boundaries
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        pip_w = self.pip_frame.winfo_width()
        pip_h = self.pip_frame.winfo_height()
        
        x = max(0, min(x, win_w - pip_w))
        y = max(0, min(y, win_h - pip_h))
        
        self.pip_frame.place(x=x, y=y, anchor="nw", relx=0.0, rely=0.0)

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def do_pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.follow_camera = False
        self.btn_follow.config(text="Follow Target: OFF", bg="#555555")
        self.update_map_display()

    def zoom_map(self, event):
        # Handle scroll wheel zooming (Windows/macOS: delta, Linux: event.num)
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            zoom_factor = 1.15
        else:
            zoom_factor = 0.85
        
        self.draw_scale = np.clip(self.draw_scale * zoom_factor, 15.0, 500.0)
        self.update_map_display()

    def toggle_follow(self):
        self.follow_camera = not self.follow_camera
        if self.follow_camera:
            self.btn_follow.config(text="Follow Target: ON", bg="#006600")
            self.pan_offset_x = 0
            self.pan_offset_y = 0
        else:
            self.btn_follow.config(text="Follow Target: OFF", bg="#555555")
        self.update_map_display()

    def update_map_display(self, map_image=None):
        """Draws the current 2D sparse map array onto the background label."""
        if map_image is None:
            # Create a blank BGR map canvas
            map_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            
            # Determine drawing center based on camera follow state
            tx, tz = self.last_tx, self.last_tz
            
            if self.follow_camera:
                self.pan_offset_x = -int(tx * self.draw_scale)
                self.pan_offset_y = -int(-tz * self.draw_scale)
                
            cx_draw = self.center_grid + self.pan_offset_x
            cy_draw = self.center_grid + self.pan_offset_y

            # Draw axis lines
            cv2.line(map_image, (cx_draw, 0), (cx_draw, self.map_size), (40, 40, 40), 1)
            cv2.line(map_image, (0, cy_draw), (self.map_size, cy_draw), (40, 40, 40), 1)
            
            # Draw historical red map points with Hough Line wall fitting (CAD look)
            if self.all_map_points is not None and len(self.all_map_points) > 0:
                gxs = np.int32(self.all_map_points[:, 0] * self.draw_scale) + cx_draw
                gys = np.int32(-self.all_map_points[:, 1] * self.draw_scale) + cy_draw
                valid_mask = (gxs >= 0) & (gxs < self.map_size) & (gys >= 0) & (gys < self.map_size)
                
                # 1. Render binary grid to fit lines
                gray_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
                gray_map[gys[valid_mask], gxs[valid_mask]] = 255
                
                # Run OpenCV Probabilistic Hough Line Transform to group points into straight walls
                lines = cv2.HoughLinesP(gray_map, rho=1, theta=np.pi/180, threshold=12, minLineLength=20, maxLineGap=12)
                if lines is not None:
                    for line in lines:
                        # Handle shape variations from OpenCV (e.g., (1, 4) vs (4,))
                        pts = line[0] if (hasattr(line, 'shape') and len(line.shape) > 1 and line.shape[0] == 1) else line
                        if len(pts) == 4:
                            x1, y1, x2, y2 = pts
                            # Draw bold wall lines in dark red
                            cv2.line(map_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 160), 2)
                
                # 2. Draw precise 1-pixel red dots on top of the walls
                for i in range(len(gxs)):
                    if valid_mask[i]:
                        cv2.circle(map_image, (gxs[i], gys[i]), 1, (0, 0, 255), -1)
            
            # Draw historical 2D pose graph trajectory (Green lines)
            for i in range(len(self.keyframe_poses) - 1):
                p1 = self.keyframe_poses[i]
                p2 = self.keyframe_poses[i + 1]
                g1x = int(p1[0] * self.draw_scale) + cx_draw
                g1y = int(-p1[2] * self.draw_scale) + cy_draw
                g2x = int(p2[0] * self.draw_scale) + cx_draw
                g2y = int(-p2[2] * self.draw_scale) + cy_draw
                if (0 <= g1x < self.map_size and 0 <= g1y < self.map_size and 
                    0 <= g2x < self.map_size and 0 <= g2y < self.map_size):
                    cv2.line(map_image, (g1x, g1y), (g2x, g2y), (0, 255, 0), 1)

            # Draw blue historical keyframes (Camera frustums)
            for kx, ky, kz, kR in self.keyframe_poses:
                draw_camera_frustum_2d(map_image, kx, kz, kR, self.draw_scale, cx_draw, cy_draw, (255, 0, 0), 1)

            # Draw current camera frustum (Green)
            draw_camera_frustum_2d(map_image, tx, tz, self.last_R, self.draw_scale, cx_draw, cy_draw, (0, 255, 0), 2)
        
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
        self.map_pcd = o3d.geometry.PointCloud()
        self.keyframe_poses = []
        self.all_map_points = None
        self.last_tx = 0.0
        self.last_tz = 0.0
        self.last_R = np.eye(3)
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.follow_camera = True
        self.btn_follow.config(text="Follow Target: ON", bg="#006600")
        self.keyframe_data = []
        
        # Disable Save button while running
        self.btn_save.config(state=tk.DISABLED, bg="#333333")

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
                    # Transform camera points to world frame
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
                    
                    # --- LOOP CLOSURE CHECK & POSE GRAPH RELAXATION ---
                    loop_closed = False
                    if len(self.keyframe_data) > 60:
                        for past_kf in self.keyframe_data[:-50]:  # At least 50 frames ago
                            px, py, pz, pR = past_kf['pose']
                            dist = np.sqrt((tx - px)**2 + (tz - pz)**2)
                            if dist < 0.35:  # Spatial proximity (35 cm)
                                # Align current scan to past keyframe
                                R_loop, t_loop = icp_2d(past_kf['points'], src_pts_2d)
                                
                                # Verify alignment quality
                                aligned_pts = src_pts_2d.dot(R_loop.T) + t_loop.T
                                dists = np.linalg.norm(aligned_pts[:, None, :] - past_kf['points'][None, :, :], axis=-1)
                                mean_err = np.mean(np.min(dists, axis=1))
                                
                                if mean_err < 0.10:  # Loop closure threshold (10 cm)
                                    print(f"[SLAM] Loop closure detected! Aligned with Keyframe {past_kf['id']} (Error: {mean_err:.3f}m)")
                                    tx_corr = tx + t_loop[0, 0]
                                    tz_corr = tz + t_loop[1, 0]
                                    err_x = tx_corr - tx
                                    err_z = tz_corr - tz
                                    
                                    # Linear relaxation: distribute accumulated drift backward
                                    start_id = past_kf['id']
                                    end_id = len(self.keyframe_data)
                                    num_loop_kfs = end_id - start_id + 1
                                    
                                    for idx in range(start_id, len(self.keyframe_data)):
                                        factor = float(idx - start_id) / num_loop_kfs
                                        kf = self.keyframe_data[idx]
                                        kx, ky, kz, kR = kf['pose']
                                        kf['pose'] = (kx + factor * err_x, ky, kz + factor * err_z, kR)
                                        kf['points'][:, 0] += factor * err_x
                                        kf['points'][:, 1] += factor * err_z
                                    
                                    # Update current tracking state
                                    tx = tx_corr
                                    tz = tz_corr
                                    path_points[-1] = np.array([tx, -ty, -tz])
                                    loop_closed = True
                                    break

                    # Store keyframe data
                    current_kf = {
                        'id': len(self.keyframe_data),
                        'pose': (tx, ty, tz, R.copy()),
                        'points': pts_world[:, [0, 2]].copy()
                    }
                    self.keyframe_data.append(current_kf)
                    if len(self.keyframe_data) > 500:
                        self.keyframe_data.pop(0)

                    # Rebuild rendering cache from keyframe database
                    self.keyframe_poses = [kf['pose'] for kf in self.keyframe_data]
                    self.all_map_points = np.vstack([kf['points'] for kf in self.keyframe_data])
                    if len(self.all_map_points) > 50000:
                        self.all_map_points = self.all_map_points[-50000:]

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

            # Record active pose coordinates for main-thread rendering
            self.last_tx = tx
            self.last_tz = tz
            self.last_R = R.copy()

            # --- STEP 5: Update GUI Visual Elements (Thread-Safe) ---
            # Pre-render video overlay inside the background worker thread

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

            # Schedule GUI elements update on main Tkinter thread (Background thread does not render map image now)
            self.root.after(0, self.update_gui_frames, pip_frame_rgb)

        # Cleanup
        cap.release()
        client_socket.close()
        print("[SLAM Client] Worker thread stopped.")

        # Trigger thread exit callback
        self.root.after(0, self.on_slam_stopped)

    def update_gui_frames(self, pip_frame_rgb):
        """Thread-safe update of Tkinter widgets using PIL."""
        # 1. Dynamically render full-screen background map on UI thread
        self.update_map_display()

        # 2. Update top-right floating PiP frame
        pil_pip = Image.fromarray(pip_frame_rgb)
        self.tk_pip_img = ImageTk.PhotoImage(pil_pip)
        self.pip_label.config(image=self.tk_pip_img)

    def on_slam_stopped(self):
        """Callback when the background SLAM thread finishes."""
        self.running = False
        self.btn_start.config(state=tk.NORMAL, bg="#006600")
        self.btn_stop.config(state=tk.DISABLED, bg="#333333")
        self.on_source_change()  # Re-evaluate Browse button state

        # Enable the Save button if we have points in memory
        if len(self.map_pcd.points) > 0:
            self.btn_save.config(state=tk.NORMAL, bg="#0055ff")
        else:
            self.btn_save.config(state=tk.DISABLED, bg="#333333")

        # Keep previews visible, but stop video feed updates
        self.pip_label.config(image="")

    def save_map(self):
        """Saves the current accumulated 3D map from memory to a PLY file."""
        if len(self.map_pcd.points) > 0:
            os.makedirs("maps", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maps/Map1_{timestamp}.ply"
            print(f"[SLAM Client] Saving accumulated 3D map to '{filename}'...")
            success = o3d.io.write_point_cloud(filename, self.map_pcd)
            if success:
                messagebox.showinfo("Success", f"3D PLY Map successfully saved to:\n{filename}")
                self.btn_save.config(state=tk.DISABLED, bg="#333333")  # Disable after saving once
            else:
                messagebox.showerror("Error", "Failed to save the 3D PLY map.")
        else:
            messagebox.showwarning("Warning", "Map is empty, nothing to save.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SLAMApp(root)
    root.mainloop()
