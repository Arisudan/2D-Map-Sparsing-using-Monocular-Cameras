import cv2
import numpy as np

class VisualOdometry:
    """
    Pure Python Monocular Visual Odometry class using OpenCV.
    Tracks feature points using Lucas-Kanade optical flow and recovers relative motion.
    """
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0, min_features=150):
        # Camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.min_features = min_features

        # Initialize FAST feature detector for speed and reliability
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Global camera pose state (Rotation matrix and Translation vector)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)

        # Tracking state
        self.prev_frame = None
        self.prev_pts = None
        
        # Frame counter
        self.frame_idx = 0

    def detect_features(self, frame_gray):
        """Detect keypoints in a grayscale frame and return them as a float32 numpy array."""
        keypoints = self.detector.detect(frame_gray)
        if len(keypoints) == 0:
            return None
        # Convert keypoint objects to a coordinate list (N, 2)
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts

    def process_frame(self, frame_bgr):
        """
        Processes a new BGR video frame.
        Estimates the relative camera motion and updates the global pose.
        Returns (Rotation matrix, Translation vector).
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_idx += 1

        # 2. Handle first frame initialization
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_pts = self.detect_features(gray)
            if self.prev_pts is None:
                print("[MVO] Error: No features detected in first frame.")
            return self.cur_R, self.cur_t

        # 3. Handle tracking (if we have points to track)
        if self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_pts = self.detect_features(self.prev_frame)
            if self.prev_pts is None:
                print("[MVO] Warning: No features to track. Skipped frame.")
                self.prev_frame = gray
                return self.cur_R, self.cur_t

        # Track points from previous frame to current frame
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_pts, None, **self.lk_params
        )

        # Filter out points that were successfully tracked
        if next_pts is not None and status is not None:
            good_prev = self.prev_pts[status.ravel() == 1]
            good_next = next_pts[status.ravel() == 1]
        else:
            good_prev = np.array([])
            good_next = np.array([])

        # 4. If tracking quality drops, re-detect features
        if len(good_next) < self.min_features:
            print(f"[MVO] Features dropped to {len(good_next)}. Re-detecting...")
            new_pts = self.detect_features(self.prev_frame)
            if new_pts is not None:
                self.prev_pts = new_pts
                # Track again with new points
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, self.prev_pts, None, **self.lk_params
                )
                if next_pts is not None and status is not None:
                    good_prev = self.prev_pts[status.ravel() == 1]
                    good_next = next_pts[status.ravel() == 1]

        # If we still don't have enough matches, skip pose estimation for this frame
        if len(good_next) < 10:
            print("[MVO] Critical: Not enough features to track relative pose.")
            self.prev_frame = gray
            return self.cur_R, self.cur_t

        # 5. Calculate motion geometry (Essential Matrix & Pose Recovery)
        # We use RANSAC to filter out outliers (incorrectly tracked keypoints)
        E, mask = cv2.findEssentialMat(
            good_next, good_prev, 
            focal=self.fx, 
            pp=(self.cx, self.cy), 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            print("[MVO] Warning: Essential matrix estimation failed.")
            self.prev_frame = gray
            self.prev_pts = good_next
            return self.cur_R, self.cur_t

        # Recover rotation (R) and translation direction (t)
        _, R, t, mask_pose = cv2.recoverPose(
            E, good_next, good_prev, 
            focal=self.fx, 
            pp=(self.cx, self.cy), 
            mask=mask
        )

        # 6. Update global pose
        # Note: Monocular visual odometry has scale ambiguity.
        # We assume relative step size/scale is 1.0. We will scale this using depth in Phase 3.
        scale = 1.0

        # We filter out invalid motion updates (e.g. extremely large or backward jumps)
        if scale > 0.05 and t[2] > -0.9:  # Avoid backward motion singularity
            self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)

        # 7. Update tracking state for next frame
        self.prev_frame = gray
        # Only keep the inlier keypoints that were used for pose estimation
        if mask_pose is not None:
            self.prev_pts = good_next[mask_pose.ravel() > 0]
        else:
            self.prev_pts = good_next

        return self.cur_R, self.cur_t
