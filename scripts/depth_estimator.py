import os
# Force Hugging Face and Torch to download and cache models on the D drive
os.environ["HF_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/huggingface"
os.environ["TORCH_HOME"] = "d:/Drone Projects/SLAM-With-D435i-And-T265/.cache/torch"

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    """
    Wrapper class for Depth Anything V2 monocular depth estimation model.
    """
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        # Select device: GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DepthEstimator] Using device: {self.device}")
        print(f"[DepthEstimator] Loading model '{model_name}'...")
        
        # Load processor and model
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.to_pretrained(model_name) if hasattr(AutoModelForDepthEstimation, 'to_pretrained') else AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("[DepthEstimator] Model loaded successfully.")

    def predict(self, frame_bgr):
        """
        Predict depth map from an RGB image (given in BGR format).
        Returns a 2D numpy array representing the predicted depth map.
        """
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Preprocess the frame
        inputs = self.image_processor(images=frame_rgb, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract predicted depth map
        predicted_depth = outputs.predicted_depth

        # Interpolate the depth map back to the original image dimensions
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy array on CPU
        depth_map = prediction.cpu().numpy()
        return depth_map
