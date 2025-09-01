import base64
import math
import os
import cv2
import numpy as np

# Required TensorFlow import
import tensorflow as tf

# Force TensorFlow to use CPU only (disable GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Memory optimization for Render deployment (2GB RAM available)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_MEMORY_ALLOCATION'] = '0.6'  # Use 60% of available memory (1.2GB)
os.environ['OMP_NUM_THREADS'] = '2'  # Allow 2 threads for 1 CPU
os.environ['TF_CPP_VMODULE'] = 'tensorflow=0'  # Disable verbose logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable Intel optimizations
os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '0'  # Disable MKL optimizations
os.environ['TF_USE_CUDNN'] = '0'  # Disable cuDNN
os.environ['TF_USE_CUDA'] = '0'  # Disable CUDA

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class StartGameRequest(BaseModel):
    mode: str = "501"
    players: list[str] = ["Player 1"]
import globals
from game_logic import GameState

# ===============================
# --- Load TensorFlow Dart Model
# ===============================
MODEL_DIR = "models/saved_model"  # Updated path to your model location

try:
    # Memory optimization: load model with memory limits
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
    
    # Set memory limits before loading
    try:
        cpu_devices = tf.config.list_physical_devices('CPU')
        if cpu_devices:
            tf.config.experimental.set_memory_growth(cpu_devices[0], False)
    except:
        pass
    
    # Load model with memory optimization
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures["serving_default"]
    
    # Clear any unnecessary variables
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("✅ TensorFlow model loaded successfully from", MODEL_DIR)
    print("🔧 TensorFlow configured for CPU-only usage")
    print("💾 Memory optimization applied for Render deployment")
    print("🧠 Memory limit: 60% of available RAM (1.2GB)")
except Exception as e:
    print(f"❌ Failed to load TensorFlow model: {e}")
    print(f"❌ Model path: {MODEL_DIR}")
    print("❌ Please ensure the model files exist and TensorFlow is properly installed")
    raise RuntimeError(f"TensorFlow model loading failed: {e}")

CONFIDENCE_THRESHOLD = 0.85
DART_CLASS_ID = 1
MAX_DARTS = 1

LABEL_MAP = {1: "dart", 2: "dartboard"}



# ===============================
# --- Globals (board + darts)
# ===============================
last_scoring_map = None
last_masks_dict = None
last_bull_info = None
last_warped_img = None
last_masks_rg = None
last_scores_order = None
last_transform = None
last_warp_size = None
last_warped_dart_img = None

dart_history = []   # all darts across match
turn_darts = []     # darts this turn only

# Store temporary candidates before confirming them
dart_candidates = []  # [(wx, wy, score, frame_count)]

# ===============================
# --- Game State Management
# ===============================
current_game: GameState = None

# ===============================
# --- Dart Tip Finder
# ===============================
def find_dart_tip(x1, y1, x2, y2, image, debug=False):
    """
    Find the tip of a dart within the bounding box.
    Returns the tip coordinates or None if no valid tip is detected.
    """
    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        if debug:
            print(f"⚠️ Empty ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=max(8, int(0.3 * max(roi.shape))),
                            maxLineGap=10)
    
    tip_img_coord = None
    if lines is not None and len(lines) > 0:
        # Find the longest line (most likely to be the dart shaft)
        best_line = max(lines[:,0], key=lambda l: np.hypot(l[2]-l[0], l[3]-l[1]))
        x3, y3, x4, y4 = best_line
        
        # Choose the tip end (usually the smaller x coordinate for horizontal darts)
        chosen = (x3, y3) if x3 < x4 else (x4, y4)
        tip_img_coord = (chosen[0] + x1, chosen[1] + y1)
        
        if debug:
            print(f"✅ Dart tip detected: {tip_img_coord} from line {best_line}")
    else:
        if debug:
            print(f"⚠️ No lines detected in ROI for dart tip detection")
    
    # 🚀 REMOVED FORCED FALLBACK: Only return detected tip or None
    # This prevents false positive dart tip detection and improves accuracy
    return tip_img_coord

# ===============================
# --- Dart Detector Wrapper
# ===============================
def run_detector(image_bgr, debug=False):
    try:
        h_orig, w_orig, _ = image_bgr.shape
        image_resized = cv2.resize(image_bgr, (640, 640))
        input_tensor = tf.convert_to_tensor(image_resized)[tf.newaxis, ...]
        input_tensor = tf.cast(input_tensor, tf.uint8)

        outputs = infer(input_tensor)
        boxes = outputs["detection_boxes"][0].numpy()   # normalized [ymin, xmin, ymax, xmax]
        scores = outputs["detection_scores"][0].numpy()
        classes = outputs["detection_classes"][0].numpy().astype(int)

        # Clear input tensor to free memory
        del input_tensor

        # --- Apply TensorFlow Non-Max Suppression ---
        selected_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=MAX_DARTS,   # limit detections (e.g., 3 darts max)
            iou_threshold=0.5,           # discard if boxes overlap > 30%
            score_threshold=CONFIDENCE_THRESHOLD  # minimum confidence
        ).numpy()

        results = []
        for i in selected_indices:
            if classes[i] != DART_CLASS_ID:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w_orig), int(ymin * h_orig)
            x2, y2 = int(xmax * w_orig), int(ymax * h_orig)

            tip_coords = find_dart_tip(x1, y1, x2, y2, image_bgr, debug)

            if tip_coords is not None:
                tip_x, tip_y = tip_coords
                results.append((x1, y1, x2, y2, scores[i], tip_x, tip_y))
                if debug:
                    print(f"✅ Valid dart with tip: score={scores[i]:.2f}, tip=({tip_x}, {tip_y})")
            else:
                if debug:
                    print(f"⚠️ Dart detected but tip not found - skipping: score={scores[i]:.2f}")

        if MAX_DARTS == 1 and results:
            return [results[0]]
        return results

    except Exception as e:
        print(f"❌ TensorFlow detection failed: {e}")
        raise RuntimeError(f"Dart detection failed: {e}")
    finally:
        import gc
        gc.collect()
        tf.keras.backend.clear_session()

def get_red_mask(hsv):
    """Binary mask for red regions (covers hue wrap-around)."""
    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )

def get_ring_mask(hsv):
    """Binary mask for double ring colors (red + green)."""
    mask_red = get_red_mask(hsv)
    lower_green, upper_green = np.array([35, 50, 50]), np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.bitwise_or(mask_red, mask_green)


# -------------------------------
# Color Filter: Detect Double Ring & Warp to Scoring Region
# -------------------------------
def warp_to_scoring_region(image, size=640):
    """
    Detect double ring (red+green), fit ellipse, and warp to a centered circle.
    Now also returns the perspective matrix M (coarse_crop -> warped scoring space).
    Returns: (warped, box, ellipse, M)
    """
    if image is None or image.size == 0:
        return None, None, None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- red+green mask ---
    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])
    lower_green, upper_green = np.array([35, 50, 50]), np.array([90, 255, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    ring_mask = cv2.bitwise_or(mask_red, mask_green)

    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, None

    largest = max(cnts, key=cv2.contourArea)
    if len(largest) < 5:
        return None, None, None, None

    # Fit ellipse just for center/size reference
    ellipse = cv2.fitEllipse(largest)  # ((cx,cy),(MA,ma),angle)

    # Use minAreaRect for perspective correction
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype("float32")

    # Destination points: square
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    # Order box points: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],        # top-left
            pts[np.argmin(diff)],     # top-right
            pts[np.argmax(s)],        # bottom-right
            pts[np.argmax(diff)]      # bottom-left
        ], dtype="float32")

    src = order_points(box)

    M = cv2.getPerspectiveTransform(src, dst)  # coarse_crop -> warped 640x640
    warped = cv2.warpPerspective(image, M, (size, size))

    # Mask into a circle
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
    warped = cv2.bitwise_and(warped, warped, mask=mask)

    return warped, box, ellipse, M

# -------------------------------
# TensorFlow Dartboard Detection (coarse crop)
# -------------------------------
def detect_dartboard_tf(image, width=640, height=640):
    try:
        input_tensor = np.expand_dims(image, axis=0).astype(np.uint8)
        preds = infer(tf.constant(input_tensor))

        boxes = preds["detection_boxes"][0].numpy()
        scores = preds["detection_scores"][0].numpy()
        classes = preds["detection_classes"][0].numpy().astype(int)

        h, w = image.shape[:2]
        for i, score in enumerate(scores):
            if score < 0.5:
                continue

            class_id = classes[i]
            label = LABEL_MAP.get(class_id, "unknown")

            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = (
                int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            )

            if label == "dartboard":
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (width, height))
                return crop, (x1, y1, x2, y2), float(score)

        return None, None, None

    except Exception as e:
        print("TF Detection error:", e)
        raise RuntimeError(f"Dartboard detection failed: {e}")
    

# -------------------------------
# HSV Fallback (coarse crop)
# -------------------------------
def detect_dartboard_hsv(image, width=640, height=640):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ring_mask = get_ring_mask(hsv)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Expand bounding box by 10% to be safe
    pad_w, pad_h = int(0.1 * w), int(0.1 * h)
    x, y = max(0, x - pad_w), max(0, y - pad_h)
    w, h = min(image.shape[1] - x, w + 2 * pad_w), min(image.shape[0] - y, h + 2 * pad_h)

    # Simple axis-aligned warp (scale)
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return None, None, None
    warped = cv2.resize(crop, (width, height))
    return warped, (x, y, x + w, y + h), 0.4  # pseudo-confidence for fallback

# -------------------------------
# Bullseye Detection (for wedge labeling center)
# -------------------------------
def detect_bullseye(img, center_hint=None):
    """
    Find inner/outer bull using red mask, preferring blobs near center_hint.
    Returns: (image, (cx,cy), r)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )

    h, w = mask.shape
    if center_hint is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = int(center_hint[0]), int(center_hint[1])

    # Emphasize inner region around the expected center
    inner_r = int(0.28 * min(cx, cy, w - cx, h - cy))
    if inner_r > 0:
        inner = np.zeros_like(mask)
        cv2.circle(inner, (cx, cy), inner_r, 255, -1)
        mask = cv2.bitwise_and(mask, inner)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, (cx, cy), int(0.06 * (min(h, w) // 2))

    # Choose blob closest to center (small penalty for size)
    candidates = []
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        d = np.hypot(x - cx, y - cy)
        candidates.append((d + 0.5 * r, (int(x), int(y)), int(r)))
    candidates.sort(key=lambda t: t[0])
    _, center, r = candidates[0]
    return img, center, r

def detect_top_wedge_boundaries(image, center, radius, red_mask):
    """
    Detect the left and right angles of the topmost red region (double 20).
    Returns (theta_left, theta_right).
    """
    cx, cy = center

    # --- Step 1: find red contours ---
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_contour = None
    top_y = 1e9
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # ignore noise
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cy_cnt = int(M["m01"] / M["m00"])
        if cy_cnt < top_y:  # smallest y = topmost
            top_y = cy_cnt
            top_contour = cnt

    if top_contour is None:
        print("⚠ No red wedge detected, fallback to default")
        return -np.pi/2 - (np.pi/20), -np.pi/2 + (np.pi/20)  # 20 wedge approx top

    # --- Step 2: get extreme left/right points of top red contour ---
    leftmost = tuple(top_contour[top_contour[:,:,0].argmin()][0])
    rightmost = tuple(top_contour[top_contour[:,:,0].argmax()][0])

    # Convert to angles (relative to board center)
    theta_left = np.arctan2(leftmost[1]-cy, leftmost[0]-cx)
    theta_right = np.arctan2(rightmost[1]-cy, rightmost[0]-cx)

    # Normalize so right is always greater than left
    if theta_right < theta_left:
        theta_right += 2*np.pi

    return theta_left, theta_right

# -------------------------------
# Wedge Detection (draw solid sector lines + labels)
# -------------------------------
def draw_wedges_aligned(scoring_img, bull_center, alpha=0.4):
    h, w = scoring_img.shape[:2]
    cx, cy = bull_center if bull_center else (w // 2, h // 2)

    # Extend lines to the very edge (double ring)
    radius = int(2 * min(cx, cy, w - cx, h - cy))

    scores_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                    3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    vis = scoring_img.copy()
    vis_clean = scoring_img.copy()  # <-- will stay clean
    masks_dict = {}
    ang_step = 2 * np.pi / 20  # radians per wedge

    # ----------------------------
    # Build HSV masks (only once)
    # ----------------------------
    hsv = cv2.cvtColor(scoring_img, cv2.COLOR_BGR2HSV)

    # Red regions (double ring, single ring, etc.)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # Black regions (alternate wedges)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 80, 80))

    # Combined mask for centroid analysis
    combined_mask = cv2.bitwise_or(black_mask, red_mask)

    # 🔑 Get top wedge boundaries (angles in radians)
    left_angle, right_angle = detect_top_wedge_boundaries(scoring_img, (cx, cy), radius, red_mask)

    # Center of top wedge
    top_angle = (left_angle + right_angle) / 2.0
    # Align wedges so that top wedge is score 20
    start_angle = top_angle - (ang_step / 2.0)

    # ----------------------------
    # Detect actual wedge color centroid to rotate scores_order
    # ----------------------------
    '''
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        top_cnt = min(contours, key=lambda c: cv2.boundingRect(c)[1])
        M = cv2.moments(top_cnt)
        if M["m00"] > 0:
            tx = int(M["m10"] / M["m00"])
            ty = int(M["m01"] / M["m00"])
            dx, dy = tx - cx, ty - cy
            cnt_angle = np.arctan2(dy, dx)  # radians
            # Find wedge index from center
            idx = int(((cnt_angle - start_angle) % (2*np.pi)) / ang_step)
            while scores_order[idx] != 20:
                scores_order = scores_order[1:] + scores_order[:1]
                '''
    # ----------------------------

    # Draw wedges + transparent lines
    overlay = vis.copy()  # for transparent line drawing
    for i, score in enumerate(scores_order):
        start = start_angle + i * ang_step
        end = start + ang_step

        # Create wedge mask
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (cx, cy), (radius, radius), 0,
                    np.degrees(start), np.degrees(end), 255, -1)
        masks_dict[score] = mask

        # Draw wedge boundary lines on overlay
        ex = int(cx + radius * np.cos(start))
        ey = int(cy + radius * np.sin(start))
        cv2.line(overlay, (cx, cy), (ex, ey), (255, 0, 0), 1, cv2.LINE_AA)

    # Blend overlay with vis for semi-transparent lines
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    return vis, vis_clean, scores_order, masks_dict

# ------------------------------------------------
# Generate binary masks for each scoring region
# ------------------------------------------------
def classify_score_with_wedges(
    x, y, cx, cy, radius, masks_dict, warped_img,
    mask_red=None, mask_green=None
):
    """
    Compute final dart score using wedge masks + color filtering.
    Bulls, doubles, and triples are validated with color masks.
    """
    x, y = int(x), int(y)
    h, w, _ = warped_img.shape
    if not (0 <= x < w and 0 <= y < h):
        return 0

    # --- Distance from center ---
    dist = math.hypot(x - cx, y - cy)
    r_norm = dist / radius if radius > 0 else 1.0

    # --- Bulls override wedges (mask + radius) ---
    if r_norm < 0.1:  
        if mask_red is not None and 0 <= y < mask_red.shape[0] and 0 <= x < mask_red.shape[1]:
            if mask_red[y, x] > 0:
                return 50
        if mask_green is not None and 0 <= y < mask_green.shape[0] and 0 <= x < mask_green.shape[1]:
            if mask_green[y, x] > 0:
                return 25

    # --- Otherwise, wedge score ---
    wedge_score = 0
    for score, mask in masks_dict.items():
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] > 0:
                wedge_score = int(score)
                break
    if wedge_score == 0:
        return 0  # miss

    # --- Define approximate bands ---
    triple_band = (0.15, 0.37)
    double_band = (0.38, 1.00)

    # --- Apply multiplier logic with color validation ---
    if double_band[0] <= r_norm <= double_band[1]:
        if (mask_red is not None and mask_red[y, x] > 0) or \
           (mask_green is not None and mask_green[y, x] > 0):
            return wedge_score * 2
        else:
            return wedge_score

    elif triple_band[0] <= r_norm <= triple_band[1]:
        if (mask_red is not None and mask_red[y, x] > 0) or \
           (mask_green is not None and mask_green[y, x] > 0):
            return wedge_score * 3
        else:
            return wedge_score

    # --- Default single ---
    return wedge_score

def make_color_masks(warped_img):
    """Return red/green masks for scoring detection."""
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    RED1 = ((0, 70, 50), (10, 255, 255))
    RED2 = ((170, 70, 50), (180, 255, 255))
    GREEN = ((35, 40, 40), (85, 255, 255))

    mask_red1 = cv2.inRange(hsv, np.array(RED1[0]), np.array(RED1[1]))
    mask_red2 = cv2.inRange(hsv, np.array(RED2[0]), np.array(RED2[1]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, np.array(GREEN[0]), np.array(GREEN[1]))

    return mask_red, mask_green
def process_dartboard(image):
    coarse_crop, coarse_box, conf = detect_dartboard_tf(image)
    method = "tf"
    if coarse_crop is None:
        coarse_crop, coarse_box, conf = detect_dartboard_hsv(image)
        method = "hsv_bbox" if coarse_crop is not None else None
    if coarse_crop is None:
        return None, None, None, None, None, None, None, None, None, None

    warped, _, ellipse, M_crop2warp = warp_to_scoring_region(coarse_crop, size=640)

    if warped is not None:
        scoring_img = warped
        scoring_method = f"{method}+double_ring_warp"
        center_hint = (scoring_img.shape[1]//2, scoring_img.shape[0]//2)
    else:
        scoring_img = coarse_crop.copy()
        scoring_method = f"{method}_no_scoring_warp"
        if ellipse is not None:
            center_hint = (int(ellipse[0][0]), int(ellipse[0][1]))
        else:
            center_hint = (scoring_img.shape[1]//2, scoring_img.shape[0]//2)
        M_crop2warp = None

    scoring_box_global = coarse_box
    _, bull_center, _ = detect_bullseye(scoring_img, center_hint=center_hint)
    wedges_vis, wedges_clean, scores_order, masks_dict = draw_wedges_aligned(scoring_img, bull_center)

    radius = int(2 * min(
        bull_center[0], bull_center[1],
        scoring_img.shape[1]-bull_center[0],
        scoring_img.shape[0]-bull_center[1]
    ))
    mask_red, mask_green = make_color_masks(wedges_clean)

    # --- Compose transform: full image -> warped scoring space
    if M_crop2warp is not None:
        x, y, w, h = coarse_box
        # shift crop back to global
        M_offset = np.array([
            [1, 0, -x],
            [0, 1, -y],
            [0, 0, 1]
        ], dtype=np.float32)
        M_full = M_crop2warp @ M_offset
    else:
        M_full = None

    return (wedges_vis, scores_order, scoring_box_global, conf,
            scoring_method, masks_dict, (bull_center, radius),
            (mask_red, mask_green), scoring_img, M_full)

# ===============================
# --- FastAPI App
# ===============================
app = FastAPI(title="Bullseye API", version="2.0.0")

# Add CORS middleware to allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    port = os.environ.get('PORT', '8000')
    print("🚀 Bullseye API starting up...")
    print(f"🌐 Host: 0.0.0.0")
    print(f"🔌 Port: {port}")
    print(f"💾 Memory limit: 60% of available RAM (1.2GB)")
    print(f"🔧 TensorFlow CPU-only mode")
    print("✅ API ready to receive requests on port " + str(port))

@app.post("/init-board")
async def init_board(file: UploadFile = File(...)):
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # --- Run dartboard detection ---
    result = process_dartboard(image)
    if result[0] is None:
        return JSONResponse({"error": "Dartboard not found"}, status_code=404)

    (
        wedges_vis, scores_order, scoring_box_global, conf,
        scoring_method, masks_dict, bull_info,
        masks_rg, warped_img, M_full
    ) = result

    bull_center, radius = bull_info
    mask_red, mask_green = masks_rg

    # --- Build scoring map (pixel → score) ---
    h, w = wedges_vis.shape[:2]
    scoring_map_full = np.zeros((h, w), np.int32)

    ys, xs = np.indices((h, w))
    for x, y in zip(xs.flatten(), ys.flatten()):
        try:
            score_val = classify_score_with_wedges(
                int(x), int(y),
                int(bull_center[0]), int(bull_center[1]),
                int(radius),
                masks_dict,
                warped_img,
                mask_red=mask_red,
                mask_green=mask_green
            )
        except Exception:
            score_val = 0
        scoring_map_full[y, x] = score_val

    # --- Save globals for later use ---
    global last_scoring_map, last_scoring_shape
    global last_masks_dict, last_bull_info, last_warped_img
    global last_masks_rg, last_scores_order, last_transform, last_warp_size
    global last_warped_dart_img, dart_history, turn_darts

    last_scoring_map = scoring_map_full.copy()
    last_scoring_shape = scoring_map_full.shape
    last_masks_dict = masks_dict.copy()
    last_bull_info = bull_info
    last_warped_img = warped_img.copy()
    last_masks_rg = (mask_red, mask_green)
    last_scores_order = scores_order
    last_transform = M_full
    last_warp_size = (w, h)
    last_warped_dart_img = None
    dart_history = []
    turn_darts = []

    return {
        "status": "dartboard initialized",
        "detected": True,
        "confidence": conf,
        "bbox": scoring_box_global,
        "method": scoring_method,
        "scores_order": scores_order,
        "scoring_map_shape": scoring_map_full.shape,
        "scoring_map_dtype": str(scoring_map_full.dtype),
        "bull_info": {
            "center": [int(bull_center[0]), int(bull_center[1])],
            "radius": int(radius)
        }
    }

@app.post("/detect-dart")
async def detect_dart(file: UploadFile = File(...)):
    global dart_history, turn_darts, current_game
    global last_transform, last_warp_size, last_scoring_map
    global last_warped_img, last_masks_dict, last_bull_info, last_masks_rg
    global last_warped_dart_img

    if last_transform is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # --- Detect darts on the ORIGINAL image ---
    detections = run_detector(image, debug=False)  # Can be enabled for debugging
    if not detections:
        return JSONResponse({"error": "No dart detected"}, status_code=404)

    # Work on a copy of the clean warped board
    vis_img = last_warped_img.copy()

    new_darts = []
    h, w = last_warp_size

    # Invert the homography: image → warped board
    inv_transform = np.linalg.inv(last_transform)

    for (x1, y1, x2, y2, conf, tip_x, tip_y) in detections:
        # Project tip into warped coordinates
        pt = np.array([[[tip_x, tip_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pt, inv_transform)[0][0]
        wx, wy = int(np.clip(warped_pt[0], 0, w - 1)), int(np.clip(warped_pt[1], 0, h - 1))

        dart_score = int(last_scoring_map[wy, wx])

        dart_entry = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "tip": [int(tip_x), int(tip_y)],   # original coords
            "tip_warped": [wx, wy],            # warped coords
            "x": wx,
            "y": wy,
            "score": dart_score
        }

        dart_history.append(dart_entry)
        turn_darts.append(dart_entry)
        new_darts.append(dart_entry)
        
        # 🎯 INTEGRATE WITH GAMESTATE: Add dart to game logic
        if current_game is not None:
            # Determine multiplier based on dart position (simplified logic)
            multiplier = 1  # Default to single
            if dart_score == 50:  # Bullseye
                multiplier = 1
            elif dart_score > 20:  # Triple ring (outer ring)
                multiplier = 3
            elif dart_score > 10:  # Double ring (middle ring)
                multiplier = 2
            else:  # Single ring (inner ring)
                multiplier = 1
            
            # Add dart to game state
            result = current_game.add_dart(dart_score, multiplier)
            print(f"🎯 Dart added to game: {dart_score} x{multiplier} = {result}")

        # --- Draw visualization on warped board ---
        cv2.circle(vis_img, (wx, wy), 8, (0, 0, 255), -1)  # red dot
        cv2.putText(
            vis_img, str(dart_score),
            (wx + 10, wy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

    last_warped_dart_img = vis_img.copy()

    # Encode visualization to base64
    _, buf = cv2.imencode(".png", vis_img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "new_darts": new_darts,
        "turn_darts": turn_darts,
        "all_darts": dart_history,
        "turn_total": int(sum(d["score"] for d in turn_darts)),
        "visualization": img_b64
    }


@app.post("/detect-dart-debug")
async def detect_dart_debug(file: UploadFile = File(...)):
    """Debug endpoint for dart detection with detailed logging."""
    global last_transform, last_warp_size, last_scoring_map
    
    if last_transform is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Run detector with debug enabled
    detections = run_detector(image, debug=True)
    
    if not detections:
        return {
            "status": "no_darts",
            "message": "No darts detected in frame (debug mode)",
            "darts": [],
            "debug_info": "No valid dart tips found"
        }
    
    # Process detections with debug info
    debug_darts = []
    for (x1, y1, x2, y2, conf, tip_x, tip_y) in detections:
        debug_darts.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "tip": [int(tip_x), int(tip_y)],
            "debug_note": "Valid tip detected"
        })
    
    return {
        "status": "success",
        "darts": debug_darts,
        "message": f"Detected {len(debug_darts)} dart(s) with valid tips",
        "debug_mode": True,
        "total_detections": len(debug_darts)
    }

@app.post("/reset-turn")
async def reset_turn():
    global turn_darts, current_game
    turn_darts = []
    
    if current_game is not None:
        # Reset turn in game state
        current_game.end_turn(force=True)
        print("🔄 Turn reset in game state")
    
    return {"status": "turn reset"}

@app.get("/debug-visual")
async def debug_visual():
    global dart_history, last_warped_img, last_warped_dart_img, last_warp_size

    if last_warped_img is None or last_warp_size is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)

    images_out = {}

    # --- Clean warped board (reference) ---
    _, buf_clean = cv2.imencode(".png", last_warped_img)
    images_out["clean_image_b64"] = base64.b64encode(buf_clean).decode("utf-8")

    if last_warped_dart_img is not None:
        # --- Dart overlay on warped dart photo ---
        vis = last_warped_dart_img.copy()
        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)]

        for i, d in enumerate(dart_history):
            if "tip_warped" not in d:
                continue
            wx, wy = map(int, d["tip_warped"])
            color = colors[i % len(colors)]

            cv2.circle(vis, (wx, wy), 10, color, -1)
            if "bbox" in d:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                vis, str(d.get("score", "?")), (wx + 12, wy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )

        _, buf_darts = cv2.imencode(".png", vis)
        images_out["darts_image_b64"] = base64.b64encode(buf_darts).decode("utf-8")

        # --- Side-by-side comparison ---
        h = max(last_warp_size[1], vis.shape[0])
        w = last_warp_size[0] * 2
        side = np.zeros((h, w, 3), dtype=np.uint8)

        side[:last_warped_img.shape[0], :last_warped_img.shape[1]] = last_warped_img
        side[:vis.shape[0], last_warped_img.shape[1]:last_warped_img.shape[1]+vis.shape[1]] = vis

        _, buf_side = cv2.imencode(".png", side)
        images_out["comparison_image_b64"] = base64.b64encode(buf_side).decode("utf-8")

    return {
        "status": "ok",
        "dart_count": len(dart_history),
        **images_out
    }

# ===============================
# --- Live Tracking Endpoints
# ===============================

@app.post("/start-live-tracking")
async def start_live_tracking():
    """Start live tracking mode for automatic dart detection."""
    global dart_history, turn_darts
    
    # Reset tracking state
    dart_history = []
    turn_darts = []
    
    return {
        "status": "success",
        "message": "Live tracking started successfully",
        "tracking_active": True
    }

@app.post("/stop-live-tracking")
async def stop_live_tracking():
    """Stop live tracking mode."""
    return {
        "status": "success",
        "message": "Live tracking stopped successfully",
        "tracking_active": False
    }

@app.post("/live-dart-detect")
async def live_dart_detect(file: UploadFile = File(...)):
    """Process a frame for live dart detection."""
    global dart_history, turn_darts, current_game, last_transform, last_warp_size, last_scoring_map
    global last_warped_img, last_masks_dict, last_bull_info, last_masks_rg
    
    if last_transform is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Detect darts using existing detection logic
    detections = run_detector(image, debug=False)  # Can be enabled for debugging
    if not detections:
        return {
            "status": "no_darts",
            "message": "No darts detected in frame",
            "darts": [],
            "game_update": None
        }
    
    # Process detections with deduplication
    new_darts = []
    h, w = last_warp_size
    
    # Invert the homography: image → warped board
    inv_transform = np.linalg.inv(last_transform)
    
    for (x1, y1, x2, y2, conf, tip_x, tip_y) in detections:
        # Project tip into warped coordinates
        pt = np.array([[[tip_x, tip_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pt, inv_transform)[0][0]
        wx, wy = int(np.clip(warped_pt[0], 0, w - 1)), int(np.clip(warped_pt[1], 0, h - 1))
        
        dart_score = int(last_scoring_map[wy, wx])
        
        # 🚫 DEDUPLICATION + TEMPORAL FILTER
        is_duplicate = False
        recent_darts = dart_history[-5:] if len(dart_history) > 5 else dart_history

        for existing_dart in recent_darts:
            distance = ((wx - existing_dart['x']) ** 2 + (wy - existing_dart['y']) ** 2) ** 0.5
            if distance < 20.0 and dart_score == existing_dart['score']:  # increased from 10 → 20
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # --- Temporal candidate filter ---
        confirmed = False
        for candidate in dart_candidates:
            cx, cy, cscore, frame_count = candidate
            distance = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
            if distance < 20.0 and dart_score == cscore:  # match threshold also 20px
                candidate[3] += 1  # increment frame count
                if candidate[3] >= 3:  # ✅ must appear in 3 consecutive frames
                    confirmed = True
                    dart_candidates.remove(candidate)
                break

        if not confirmed:
            # Add or update candidate
            dart_candidates.append([wx, wy, dart_score, 1])
            continue  # wait for stability before adding to history

        # Check if we already have too many darts in this turn (max 3)
        if len(turn_darts) >= 3:
            print(f"⚠️ Turn already has 3 darts, skipping new detection")
            continue
        
        dart_entry = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "tip": [int(tip_x), int(tip_y)],
            "tip_warped": [wx, wy],
            "x": wx,
            "y": wy,
            "score": dart_score
        }
        
        dart_history.append(dart_entry)
        turn_darts.append(dart_entry)
        new_darts.append(dart_entry)
        
        print(f"✅ New dart detected: {dart_score} points at position ({wx}, {wy})")
        
        # 🎯 INTEGRATE WITH GAMESTATE: Add dart to game logic
        if current_game is not None:
            # Determine multiplier based on dart position (simplified logic)
            multiplier = 1  # Default to single
            if dart_score == 50:  # Bullseye
                multiplier = 1
            elif dart_score > 20:  # Triple ring (outer ring)
                multiplier = 3
            elif dart_score > 10:  # Double ring (middle ring)
                multiplier = 2
            else:  # Single ring (inner ring)
                multiplier = 1
            
            # Add dart to game state
            result = current_game.add_dart(dart_score, multiplier)
            print(f"🎯 Live dart added to game: {dart_score} x{multiplier} = {result}")
    
    # Only return darts if we actually detected some
    if new_darts:
        # Get current game state if available
        game_update = None
        if current_game is not None:
            game_update = current_game.get_state()
        
        return {
            "status": "success",
            "darts": new_darts,
            "message": f"Detected {len(new_darts)} new dart(s)",
            "total_darts": len(dart_history),
            "turn_darts": len(turn_darts),
            "game_update": game_update
        }
    else:
        return {
            "status": "no_darts",
            "message": "No new darts detected",
            "darts": [],
            "total_darts": len(dart_history),
            "turn_darts": len(turn_darts),
            "game_update": None
        }

class GameStartRequest(BaseModel):
    mode: str = "501"
    players: list = ["Player 1"]

@app.post("/start-game")
async def start_game(request: GameStartRequest):
    """Start a new game with specified mode and players."""
    global dart_history, turn_darts, current_game
    
    # Create new game instance with real GameState
    current_game = GameState(
        mode=request.mode,
        players=request.players,
        double_out=True
    )
    
    # Reset game state
    dart_history = []
    turn_darts = []
    
    return {
        "status": "success",
        "message": "Game started successfully",
        "game_state": current_game.get_state()
    }

@app.post("/end-turn")
async def end_turn():
    """End the current turn and calculate final scores."""
    global turn_darts, current_game
    
    if current_game is None:
        return JSONResponse({"error": "No game in progress"}, status_code=400)
    
    # End turn using real GameState
    current_game.end_turn()
    
    # Get updated game state
    game_state = current_game.get_state()
    
    return {
        "status": "success",
        "message": "Turn ended successfully",
        "turn_total": sum(d["score"] for d in turn_darts),
        "game_state": game_state
    }

@app.get("/game-state")
async def get_game_state():
    """Get current game state."""
    global current_game
    
    if current_game is None:
        return JSONResponse({"error": "No game in progress"}, status_code=400)
    
    # Get real game state from GameState instance
    game_state = current_game.get_state()
    
    return {
        "status": "success",
        "game_state": game_state
    }

@app.get("/current-score")
async def get_current_score():
    """Get current player's remaining score."""
    global current_game
    
    if current_game is None:
        return JSONResponse({"error": "No game in progress"}, status_code=400)
    
    current_player = current_game.players[current_game.current_player]
    remaining_score = current_game.scores[current_player]
    
    return {
        "status": "success",
        "current_player": current_player,
        "remaining_score": remaining_score,
        "turn_darts": len(current_game.turn_darts),
        "game_mode": current_game.mode
    }

# ===============================
# --- Run
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Bullseye API on port {port}")
    print(f"🌐 Host: 0.0.0.0")
    print(f"🔌 Port: {port}")
    print(f"💾 Memory limit: 20% of available RAM")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)

@app.get("/")
def root():
    return {
        "status": "Bullseye API is running",
        "version": "2.0.0",
        "model_loaded": True,
        "model_path": MODEL_DIR,
        "endpoints": [
            "/init-board (POST)",
            "/detect-dart (POST)", 
            "/detect-dart-debug (POST)",  # 🆕 Debug endpoint
            "/create-yellow-circle-overlay (POST)",  # 🆕 Yellow circle overlay
            "/reset-turn (POST)",
            "/debug-visual (GET)",
            "/healthz (GET)",
            "/start-live-tracking (POST)",
            "/stop-live-tracking (POST)",
            "/live-dart-detect (POST)",
            "/start-game (POST)",
            "/end-turn (POST)",
            "/game-state (GET)",
            "/current-score (GET)",
            "/board-overlay (GET)",
            "/board-overlay-visual (GET)",
            "/debug-board (GET)"
        ]
    }

@app.get("/debug-board")
def debug_board():
    """Debug endpoint to check board initialization status."""
    return {
        "board_initialized": last_transform is not None,
        "scoring_map_ready": last_scoring_map is not None,
        "transform_matrix": str(last_transform) if last_transform is not None else None,
        "warp_size": last_warp_size,
        "dart_history_count": len(dart_history),
        "turn_darts_count": len(turn_darts),
        "last_bull_info": last_bull_info
    }

@app.post("/create-yellow-circle-overlay")
async def create_yellow_circle_overlay(file: UploadFile = File(...)):
    """
    Create a yellow overlay that follows the actual scoring zone boundaries of the dartboard.
    Uses detect_dartboard_tf and creates a visualizer that sticks to the scoring region outline.
    """
    try:
        # Read the uploaded image
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
        
        # Step 1: Detect dartboard using detect_dartboard_tf
        print("🎯 Detecting dartboard using TensorFlow...")
        coarse_crop, coarse_box, conf = detect_dartboard_tf(image)
        
        if coarse_crop is None:
            return JSONResponse({"error": "Dartboard not detected"}, status_code=404)
        
        print(f"✅ Dartboard detected with confidence: {conf}")
        
        # Step 2: Create green and red mask detection
        print("🎨 Creating green and red masks...")
        hsv = cv2.cvtColor(coarse_crop, cv2.COLOR_BGR2HSV)
        
        # Red mask (two ranges for red color)
        lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        
        # Green mask
        lower_green, upper_green = np.array([35, 50, 50]), np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine red and green masks
        ring_mask = cv2.bitwise_or(mask_red, mask_green)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
        
        # Step 2.5: Check camera stability by analyzing mask quality
        print("📱 Checking camera stability...")
        
        # Calculate mask statistics for stability assessment
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        total_pixels = mask_red.shape[0] * mask_red.shape[1]
        
        # Calculate mask coverage percentages
        red_coverage = red_pixels / total_pixels
        green_coverage = green_pixels / total_pixels
        combined_coverage = (red_pixels + green_pixels) / total_pixels
        
        print(f"📊 Mask coverage - Red: {red_coverage:.3f}, Green: {green_coverage:.3f}, Combined: {combined_coverage:.3f}")
        
        # Check if masks are well-defined (stable camera)
        min_red_coverage = 0.01    # At least 1% red pixels
        min_green_coverage = 0.01  # At least 1% green pixels
        min_combined_coverage = 0.02  # At least 2% combined pixels
        
        camera_stable = (
            red_coverage >= min_red_coverage and 
            green_coverage >= min_green_coverage and 
            combined_coverage >= min_combined_coverage
        )
        
        if not camera_stable:
            print("❌ Camera not stable - masks too weak")
            return JSONResponse({
                "error": "Camera not stable enough for accurate detection",
                "details": {
                    "red_coverage": red_coverage,
                    "green_coverage": green_coverage,
                    "combined_coverage": combined_coverage,
                    "min_required": {
                        "red": min_red_coverage,
                        "green": min_green_coverage,
                        "combined": min_combined_coverage
                    }
                }
            }, status_code=400)
        
        print("✅ Camera is stable - masks are well-defined")
        
        # Step 3: Find contours and fit ellipse
        print("🔍 Finding contours and fitting ellipse...")
        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return JSONResponse({"error": "No contours found in mask"}, status_code=404)
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            return JSONResponse({"error": "Contour too small for ellipse fitting"}, status_code=404)
        
        # Additional stability check: contour quality and blur detection
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
        
        # Calculate contour circularity (should be close to 1 for a good circle)
        if contour_perimeter > 0:
            circularity = 4 * np.pi * contour_area / (contour_perimeter * contour_perimeter)
        else:
            circularity = 0
        
        print(f"📐 Contour analysis - Area: {contour_area:.0f}, Perimeter: {contour_perimeter:.0f}, Circularity: {circularity:.3f}")
        
        # Check if contour is well-formed (stable camera)
        min_circularity = 0.3  # Minimum circularity for stable detection
        min_contour_area = 1000  # Minimum contour area
        
        if circularity < min_circularity or contour_area < min_contour_area:
            print("❌ Contour quality too poor - camera may be moving or blurry")
            return JSONResponse({
                "error": "Contour quality too poor for accurate detection",
                "details": {
                    "circularity": circularity,
                    "contour_area": contour_area,
                    "min_required": {
                        "circularity": min_circularity,
                        "contour_area": min_contour_area
                    }
                }
            }, status_code=400)
        
        print("✅ Contour quality is good - camera is stable")
        
        # Additional stability check: image sharpness (blur detection)
        print("🔍 Checking image sharpness...")
        
        # Convert to grayscale for blur detection
        gray_crop = cv2.cvtColor(coarse_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (higher = sharper image)
        laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        
        print(f"📸 Image sharpness (Laplacian variance): {laplacian_var:.2f}")
        
        # Check if image is sharp enough (not blurry)
        min_sharpness = 50.0  # Minimum Laplacian variance for sharp image
        
        if laplacian_var < min_sharpness:
            print("❌ Image too blurry - camera may be moving")
            return JSONResponse({
                "error": "Image too blurry for accurate detection",
                "details": {
                    "sharpness": laplacian_var,
                    "min_required": min_sharpness
                }
            }, status_code=400)
        
        print("✅ Image is sharp - camera is stable")
        
        # Step 4: Create scoring zone boundary overlay (like the green outline in screenshot)
        print("🎨 Creating scoring zone boundary overlay...")
        
        # Create a copy of the original image for visualization
        vis_image = image.copy()
        
        # Draw the detected dartboard bounding box
        if coarse_box is not None:
            x1, y1, x2, y2 = coarse_box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create the scoring zone overlay using draw_wedges_aligned approach
        # First, we need to detect the bullseye center
        bull_center, radius = detect_bullseye(coarse_crop)
        
        if bull_center is not None:
            # Convert bull_center to original image coordinates
            if coarse_box is not None:
                x1, y1, x2, y2 = coarse_box
                # Scale coordinates from cropped image back to original
                scale_x = (x2 - x1) / 640
                scale_y = (y2 - y1) / 640
                bull_center_orig = (
                    int(x1 + bull_center[0] * scale_x),
                    int(y1 + bull_center[1] * scale_y)
                )
                radius_orig = int(radius * min(scale_x, scale_y))
            else:
                bull_center_orig = bull_center
                radius_orig = radius
            
            # Create the scoring zone boundary overlay
            # This will create the yellow outline that follows the actual scoring zones
            
            # Method 1: Use the actual contour from the mask (most accurate)
            # Find the outer boundary of the scoring zones
            outer_contour = largest_contour
            
            # Draw the yellow outline following the actual scoring zone boundary
            cv2.drawContours(vis_image, [outer_contour], -1, (0, 255, 255), 3)
            
            # Method 2: Create a more refined boundary using the wedge masks
            # This creates a smoother outline that follows the scoring zones exactly
            h, w = coarse_crop.shape[:2]
            ang_step = 2 * np.pi / 20  # 20 wedges
            
            # Find the top wedge boundary for alignment
            # This ensures the overlay is properly oriented
            top_angle = -np.pi/2  # Start from top (20 wedge)
            
            # Create the scoring zone boundary points
            boundary_points = []
            for i in range(20):
                angle = top_angle + i * ang_step
                # Use the actual radius from the detected contour
                radius_actual = cv2.arcLength(outer_contour, True) / (2 * np.pi)
                x = int(bull_center_orig[0] + radius_actual * np.cos(angle))
                y = int(bull_center_orig[1] + radius_actual * np.sin(angle))
                boundary_points.append([x, y])
            
            # Draw the yellow boundary line connecting all points
            if len(boundary_points) > 2:
                boundary_points = np.array(boundary_points, dtype=np.int32)
                cv2.polylines(vis_image, [boundary_points], True, (0, 255, 255), 3)
            
            # Add the bullseye center indicator
            cv2.circle(vis_image, bull_center_orig, 5, (0, 255, 255), -1)
            
            # Add scoring zone labels (optional - for debugging)
            for i, score in enumerate(scores_order):
                angle = top_angle + i * ang_step
                x = int(bull_center_orig[0] + (radius_orig * 0.7) * np.cos(angle))
                y = int(bull_center_orig[1] + (radius_orig * 0.7) * np.sin(angle))
                
                # Ensure text is within image bounds
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.putText(
                        vis_image, str(score), (x-10, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )
        else:
            # Fallback: if bullseye detection fails, use the simple ellipse approach
            print("⚠️ Bullseye detection failed, using fallback ellipse method")
            ellipse = cv2.fitEllipse(largest_contour)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            
            # Draw the yellow ellipse
            cv2.ellipse(vis_image, ellipse, (0, 255, 255), 3)
            
            # Update variables for response
            bull_center_orig = (int(center_x), int(center_y))
            radius_orig = int((major_axis + minor_axis) / 2)
        
        # Add text labels
        cv2.putText(vis_image, f"Confidence: {conf:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Center: ({bull_center_orig[0]}, {bull_center_orig[1]})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Radius: {radius_orig}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to base64 for Flutter
        _, buffer = cv2.imencode('.png', vis_image)
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "message": "Scoring zone boundary overlay created successfully",
            "overlay_image": overlay_b64,
            "overlay_data": {
                "board_center": [bull_center_orig[0], bull_center_orig[1]],
                "board_radius": radius_orig,
                "confidence": float(conf),
                "detection_method": "tensorflow + scoring_zone_boundary",
                "scoring_zones_detected": len(scores_order) if 'scores_order' in locals() else 0,
                "stability_metrics": {
                    "camera_stable": True,
                    "red_coverage": float(red_coverage),
                    "green_coverage": float(green_coverage),
                    "combined_coverage": float(combined_coverage),
                    "contour_circularity": float(circularity),
                    "contour_area": int(contour_area),
                    "image_sharpness": float(laplacian_var)
                }
            }
        }
        
    except Exception as e:
        print(f"❌ Error creating scoring zone boundary overlay: {e}")
        return JSONResponse({"error": f"Failed to create overlay: {str(e)}"}, status_code=500)

@app.get("/board-overlay")
def get_board_overlay():
    """Get board overlay information for Flutter app visualization."""
    if last_transform is None or last_scoring_map is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)
    
    # Return board boundaries and scoring zones for overlay
    h, w = last_warp_size
    bull_center, radius = last_bull_info
    
    return {
        "status": "success",
        "overlay": {
            "board_center": [int(bull_center[0]), int(bull_center[1])],
            "board_radius": int(radius),
            "board_size": [w, h],
            "transform_matrix": str(last_transform),
            "scoring_map_shape": last_scoring_map.shape,
            "total_darts": len(dart_history),
            "total_darts": len(dart_history),
            "turn_darts": len(turn_darts)
        }
    }

@app.get("/board-overlay-visual")
def get_board_overlay_visual():
    """Get the actual visual overlay image that the API generates."""
    global last_warped_img, last_masks_dict, last_bull_info
    
    if last_warped_img is None or last_bull_info is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)
    
    try:
        # Create the overlay using your existing draw_wedges_aligned function
        bull_center, radius = last_bull_info
        
        # Generate the overlay with wedges and scoring zones
        overlay_img, _, scores_order, masks_dict = draw_wedges_aligned(
            last_warped_img, bull_center, alpha=0.6
        )
        
        # Add bullseye center indicator
        cv2.circle(overlay_img, (int(bull_center[0]), int(bull_center[1])), 5, (0, 255, 255), -1)
        
        # Add scoring zone labels
        h, w = overlay_img.shape[:2]
        ang_step = 2 * np.pi / 20
        start_angle = -np.pi/2  # Start from top (20 wedge)
        
        for i, score in enumerate(scores_order):
            angle = start_angle + i * ang_step
            x = int(bull_center[0] + (radius * 0.7) * np.cos(angle))
            y = int(bull_center[1] + (radius * 0.7) * np.sin(angle))
            
            # Ensure text is within image bounds
            if 0 <= x < w and 0 <= y < h:
                cv2.putText(
                    overlay_img, str(score), (x-10, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )
        
        # Convert to base64 for Flutter
        _, buffer = cv2.imencode('.png', overlay_img)
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "overlay_image": base64.b64encode(buffer).decode('utf-8'),
            "overlay_info": {
                "board_center": [int(bull_center[0]), int(bull_center[1])],
                "board_radius": int(radius),
                "scores_order": scores_order,
            }
        }
        
    except Exception as e:
        print(f"Error generating overlay: {e}")
        return JSONResponse({"error": f"Failed to generate overlay: {str(e)}"}, status_code=500)

@app.get("/ping")
def ping():
    return {"pong": True, "timestamp": "now"}

@app.get("/healthz")
def healthz():
    import psutil
    memory_info = psutil.virtual_memory()
    
    # Get TensorFlow memory info if available
    tf_memory = "N/A"
    try:
        import tensorflow as tf
        tf_memory = f"{tf.config.experimental.get_memory_info('CPU:0')['current'] / (1024**2):.1f} MB"
    except:
        pass
    
    return {
        "ok": True,
        "memory_usage": {
            "total": f"{memory_info.total / (1024**3):.2f} GB",
            "available": f"{memory_info.available / (1024**3):.2f} GB",
            "percent": f"{memory_info.percent:.1f}%",
            "tensorflow_memory": tf_memory
        },
        "render_status": "Memory optimized for 2GB RAM (1 CPU, 2GB)"
    }