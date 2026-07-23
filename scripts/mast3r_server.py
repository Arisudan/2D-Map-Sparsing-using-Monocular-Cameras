import socket
import struct
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import json
import os
import sys
import datetime
import lietorch

# Add MASt3R-SLAM directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../MASt3R-SLAM")))

from mast3r_slam.config import load_config, config
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
from main import run_backend

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main():
    print("=== MASt3R-SLAM distributed Server ===")
    
    # Initialize multiprocessing start method for CUDA
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set
        
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MASt3R-SLAM/config/base.yaml"))
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    load_config(config_path)
    
    # Load MASt3R model on RTX 5090
    print("Loading MASt3R neural network model weights...")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"))
    model = load_mast3r(path=model_path, device=device)
    model.share_memory()
    print("MASt3R Model loaded successfully!")

    # Start TCP Socket Server on Port 5000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(5)
    print("MASt3R-SLAM Server listening on port 5000...")
    
    manager = mp.Manager()
    
    while True:
        client_socket, client_addr = server_socket.accept()
        print(f"Client connected from {client_addr}")
        
        # Initialize variables for this session
        keyframes = None
        states = None
        tracker = None
        backend_proc = None
        
        frame_idx = 0
        h_resized, w_resized = 0, 0
        
        try:
            while True:
                # 1. Receive JPEG frame size
                len_bytes = recv_all(client_socket, 4)
                if len_bytes is None:
                    break
                img_len = struct.unpack('>I', len_bytes)[0]
                
                # 2. Receive JPEG frame bytes
                img_bytes = recv_all(client_socket, img_len)
                if img_bytes is None:
                    break
                
                # Decode JPEG image
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                    
                # Convert BGR to RGB and float [0.0, 1.0] for MASt3R
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # 3. Dynamic initialization on first frame
                if frame_idx == 0:
                    h_orig, w_orig = frame_bgr.shape[:2]
                    # Compute resized resolution matching MASt3R requirements (long edge = 512)
                    S = max(h_orig, w_orig)
                    w_res = int(round(w_orig * 512 / S))
                    h_res = int(round(h_orig * 512 / S))
                    # Crop to multiple of 16 as done in resize_img
                    cx, cy = w_res // 2, h_res // 2
                    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                    if w_res == h_res:
                        halfh = int(3 * halfw / 4)
                    w_resized = 2 * halfw
                    h_resized = 2 * halfh
                    
                    print(f"Session resolution: Resized to {w_resized}x{h_resized}")
                    
                    # Instantiate keyframes, states, and tracker
                    keyframes = SharedKeyframes(manager, h_resized, w_resized, device=device)
                    states = SharedStates(manager, h_resized, w_resized, device=device)
                    tracker = FrameTracker(model, keyframes, device)
                    
                    # Start backend optimization process
                    backend_proc = mp.Process(target=run_backend, args=(config, model, states, keyframes, None))
                    backend_proc.start()
                    
                # Get last pose or identity Sim(3)
                T_WC = (
                    lietorch.Sim3.Identity(1, device=device)
                    if frame_idx == 0
                    else states.get_frame().T_WC
                )
                
                # Create input Frame object
                frame = create_frame(frame_idx, frame_rgb, T_WC, img_size=512, device=device)
                
                # 4. Run tracking pipeline
                if frame_idx == 0:
                    # Init frame
                    X_init, C_init = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X_init, C_init)
                    keyframes.append(frame)
                    states.queue_global_optimization(0)
                    states.set_mode(Mode.TRACKING)
                    states.set_frame(frame)
                    add_new_kf = True
                else:
                    # Track frame
                    add_new_kf, match_info, try_reloc = tracker.track(frame)
                    states.set_frame(frame)
                    if add_new_kf:
                        keyframes.append(frame)
                        states.queue_global_optimization(len(keyframes) - 1)
                        
                # 5. Extract drift-free pose and 2D sparse edge points
                # Retrieve Camera-to-World Sim(3) transformation matrix
                T_matrix = frame.T_WC.matrix()[0].cpu().numpy()
                R_mat = T_matrix[:3, :3]
                tx, ty, tz = T_matrix[0, 3], T_matrix[1, 3], T_matrix[2, 3]
                
                # Transform canonical 3D points to world coordinates
                pts_world_tensor = frame.T_WC.act(frame.X_canon)
                pts_world = pts_world_tensor.cpu().numpy()  # Shape: (h_resized * w_resized, 3)
                
                # Extract Z coordinate locally to compute depth boundaries
                X_canon_np = frame.X_canon.cpu().numpy()
                X_canon_3d = X_canon_np.reshape(h_resized, w_resized, 3)
                depth_map = np.abs(X_canon_3d[:, :, 2])
                
                # Normalize depth and detect edges
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                edges = cv2.Canny(depth_norm, 15, 45)
                
                edge_y, edge_x = np.where(edges > 0)
                
                # Subsample edge coordinates to keep transmission packets small and light
                if len(edge_x) > 1000:
                    indices = np.random.choice(len(edge_x), size=1000, replace=False)
                    edge_x = edge_x[indices]
                    edge_y = edge_y[indices]
                    
                edge_world_2d = []
                if len(edge_x) > 0:
                    # Map 2D pixel index to flattened 3D world coordinate index
                    flat_indices = edge_y * w_resized + edge_x
                    pts_world_edges = pts_world[flat_indices]
                    edge_world_2d = pts_world_edges[:, [0, 2]].tolist()  # Keep X and Z
                    
                # 6. Stream pose and edge coordinates back to client
                response_data = {
                    'tx': float(tx),
                    'ty': float(ty),
                    'tz': float(tz),
                    'R': R_mat.tolist(),
                    'points_2d': edge_world_2d
                }
                
                response_bytes = json.dumps(response_data).encode('utf-8')
                client_socket.sendall(struct.pack('>I', len(response_bytes)) + response_bytes)
                
                frame_idx += 1
                
        except Exception as e:
            print(f"Exception during session: {e}")
        finally:
            client_socket.close()
            print("Client disconnected.")
            if backend_proc is not None:
                states.set_mode(Mode.TERMINATED)
                backend_proc.join()
                print("Backend optimization process stopped.")

if __name__ == "__main__":
    main()
