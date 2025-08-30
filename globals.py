import numpy as np

# ===============================
# Board state (initialized once)
# ===============================
last_scoring_map: np.ndarray | None = None      # Pixel → score map
last_scoring_shape: tuple | None = None         # Shape of scoring map
last_masks_dict: dict | None = None             # Wedge masks, ring masks
last_bull_info: tuple | None = None             # (center, radius)
last_warped_img: np.ndarray | None = None       # Clean warped dartboard image
last_masks_rg: tuple | None = None              # (red_mask, green_mask)
last_scores_order: list | None = None           # Score order around board
last_transform: np.ndarray | None = None        # Homography matrix (H)
last_warp_size: tuple | None = None             # Warp output size (w, h)
last_frame_warped_img: np.ndarray | None = None
last_frame_scoring_map: np.ndarray | None = None
last_frame_hash: int | None = None   # optional, to detect if the frame changed

# ===============================
# Dart state (resets every turn)
# ===============================
last_warped_dart_img: np.ndarray | None = None  # Warped dart image
dart_history: list = []                         # All darts so far
turn_darts: list = []               