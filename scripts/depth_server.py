import socket
import struct
import cv2
import numpy as np
import torch
from depth_estimator import DepthEstimator

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

def main():
    # 1. Initialize Depth Model on GPU (CUDA)
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"[Depth Server] Initializing Heavy model: {model_name}...")
    estimator = DepthEstimator(model_name=model_name)
    print(f"[Depth Server] Running on device: {estimator.device}")

    # 2. Setup TCP Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    port = 5000
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)
    print(f"[Depth Server] Listening for client connections on port {port}...")

    try:
        while True:
            conn, addr = server_socket.accept()
            print(f"[Depth Server] Connected to client at {addr}")

            try:
                while True:
                    # 1. Read the length of the incoming BGR frame (4-byte unsigned int)
                    length_bytes = recv_all(conn, 4)
                    if length_bytes is None:
                        break
                    length = struct.unpack('>I', length_bytes)[0]

                    # 2. Read the raw JPEG BGR frame bytes
                    img_bytes = recv_all(conn, length)
                    if img_bytes is None:
                        break

                    # 3. Decode JPEG to BGR numpy array
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        print("[Depth Server] Error: Failed to decode frame JPEG.")
                        continue

                    # 4. Perform Depth Inference on GPU
                    depth_map = estimator.predict(frame)

                    # 5. Compress the float32 depth map to 16-bit uint PNG
                    # Scaling by 1000 preserves millimeter accuracy (e.g. 2.345 meters -> 2345 value)
                    depth_uint16 = (depth_map * 1000.0).astype(np.uint16)
                    ret, depth_png = cv2.imencode('.png', depth_uint16)
                    if not ret:
                        print("[Depth Server] Error: Failed to compress depth map to PNG.")
                        continue
                    
                    depth_bytes = depth_png.tobytes()

                    # 6. Send the compressed depth map back to the client
                    # Format: 4-byte header length + raw PNG bytes
                    conn.sendall(struct.pack('>I', len(depth_bytes)) + depth_bytes)

            except ConnectionResetError:
                print("[Depth Server] Client disconnected unexpectedly.")
            finally:
                conn.close()
                print("[Depth Server] Connection closed. Waiting for new client...")

    except KeyboardInterrupt:
        print("\n[Depth Server] Stopping server.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
