import base64
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import globals

# ===============================
# --- Load TensorFlow Dart Model (Memory Optimized)
# ===============================
MODEL_DIR = "models/saved_model"

# Memory optimization: Load model only when needed
model = None
infer = None

def load_model():
    """Load TensorFlow model with memory optimization."""
    global model, infer
    if model is None:
        # Set TensorFlow memory growth to prevent OOM
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")
        
        # Load model
        model = tf.saved_model.load(MODEL_DIR)
        infer = model.signatures["serving_default"]
        print("TensorFlow model loaded successfully")
    
    return model, infer

CONFIDENCE_THRESHOLD = 0.3
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

dart_history = []   # all darts across match
turn_darts = []     # darts this turn only

# ===============================
# --- Dart Tip Finder
# ===============================
def find_dart_tip(x1, y1, x2, y2, image, debug=False):
    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return (x1 + x2) // 2, (y1 + y2) // 2

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=max(8, int(0.3 * max(roi.shape))),
                            maxLineGap=10)
    tip_img_coord = None
    if lines is not None:
        best_line = max(lines[:,0], key=lambda l: np.hypot(l[2]-l[0], l[3]-l[1]))
        x3, y3, x4, y4 = best_line
        chosen = (x3, y3) if x3 < x4 else (x4, y4)
        tip_img_coord = (chosen[0] + x1, chosen[1] + y1)
    if tip_img_coord is None:
        tip_img_coord = (x1, (y1 + y2) // 2)  # fallback
    return tip_img_coord

# ===============================
# --- Dart Detector Wrapper
# ===============================
def run_detector(image_bgr):
    h_orig, w_orig, _ = image_bgr.shape
    image_resized = cv2.resize(image_bgr, (640, 640))
    input_tensor = tf.convert_to_tensor(image_resized)[tf.newaxis, ...]
    input_tensor = tf.cast(input_tensor, tf.uint8)

    outputs = infer(input_tensor)
    boxes = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(int)

    results = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < CONFIDENCE_THRESHOLD: continue
        if cls != DART_CLASS_ID: continue

        ymin, xmin, ymax, xmax = box
        x1, y1 = int(xmin * w_orig), int(ymin * h_orig)
        x2, y2 = int(xmax * w_orig), int(ymax * h_orig)

        tip_x, tip_y = find_dart_tip(x1, y1, x2, y2, image_bgr)
        results.append((x1, y1, x2, y2, score, tip_x, tip_y))

    results.sort(key=lambda x: x[4], reverse=True)
    if MAX_DARTS == 1 and results:
        return [results[0]]
    return results

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
        return None, None, None
    

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
app = FastAPI()

# Add CORS middleware to allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
        "status": "Bullseye API is running",
        "endpoints": ["/init-board (POST)", "/detect-dart (POST)", "/reset-turn (POST)", "/debug-visual (GET)"]
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/init-board")
async def init_board(file: UploadFile = File(...)):
    # Load TensorFlow model if not already loaded
    try:
        load_model()
    except Exception as e:
        return JSONResponse({"error": f"Failed to load model: {str(e)}"}, status_code=500)
    
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
    global dart_history, turn_darts
    global last_transform, last_warp_size, last_scoring_map
    global last_warped_img, last_masks_dict, last_bull_info, last_masks_rg
    global last_warped_dart_img

    if last_transform is None:
        return JSONResponse({"error": "Board not initialized"}, status_code=400)

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # --- Detect darts on the ORIGINAL image ---
    detections = run_detector(image)
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


@app.post("/reset-turn")
async def reset_turn():
    global turn_darts
    turn_darts = []
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
# --- Run
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
