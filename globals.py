import numpy as np
from typing import Union, Optional

# ===============================
# Board state (initialized once)
# ===============================
last_scoring_map: Optional[np.ndarray] = None      # Pixel → score map
last_scoring_shape: Optional[tuple] = None         # Shape of scoring map
last_masks_dict: Optional[dict] = None             # Wedge masks, ring masks
last_bull_info: Optional[tuple] = None             # (center, radius)
last_warped_img: Optional[np.ndarray] = None       # Clean warped dartboard image
last_masks_rg: Optional[tuple] = None              # (red_mask, green_mask)
last_scores_order: Optional[list] = None           # Score order around board
last_transform: Optional[np.ndarray] = None        # Homography matrix (H)
last_warp_size: Optional[tuple] = None             # Warp output size (w, h)
last_frame_warped_img: Optional[np.ndarray] = None
last_frame_scoring_map: Optional[np.ndarray] = None
last_frame_hash: Optional[int] = None   # optional, to detect if the frame changed

# ===============================
# Dart state (resets every turn)
# ===============================
last_warped_dart_img: Optional[np.ndarray] = None  # Warped dart image
dart_history: list = []                         # All darts so far
turn_darts: list = []               