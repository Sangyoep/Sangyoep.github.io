import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from scipy.interpolate import splprep, splev

def spline_smooth_backbone(backbone, smoothing=5, num=100):
    if len(backbone) < 4:
        return backbone
    x = [p[0] for p in backbone]
    y = [p[1] for p in backbone]
    try:
        tck, _ = splprep([x, y], s=smoothing)
        u_fine = np.linspace(0, 1, num)
        x_smooth, y_smooth = splev(u_fine, tck)
        return list(zip(map(int, x_smooth), map(int, y_smooth)))
    except:
        return backbone  # Fallback



def smooth_backbone(backbone, window=5):
    if len(backbone) < window:
        return backbone

    smoothed = []
    for i in range(len(backbone)):
        w_start = max(0, i - window // 2)
        w_end   = min(len(backbone), i + window // 2 + 1)
        segment = backbone[w_start:w_end]
        avg_x = int(np.mean([p[0] for p in segment]))
        avg_y = int(np.mean([p[1] for p in segment]))
        smoothed.append((avg_x, avg_y))
    return smoothed

# -----------------------------------------------------------------------
# 1)  anchor + skeleton helpers
# -----------------------------------------------------------------------
def anchor_points_from_contour(cnt, strip=0.07):
    """
    Return two mid‑points on the contour:
      • base  = midpoint of lowest strip  (height × strip)
      • tip   = midpoint of right‑most strip (width × strip)
    """
    pts = cnt.reshape(-1, 2)                      # (x,y)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    h, w = (y_max - y_min), (x_max - x_min)
    base_strip = pts[pts[:, 1] >= y_max - strip * h]
    tip_strip  = pts[pts[:, 0] >= x_max - strip * w]

    def mid_of_strip(strip_pts):
        if len(strip_pts) < 2:
            return tuple(strip_pts[0])
        mu = strip_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(strip_pts - mu)
        dir_vec = vt[0]                           # principal axis
        t = (strip_pts - mu) @ dir_vec
        p1, p2 = mu + dir_vec * t.min(), mu + dir_vec * t.max()
        return tuple(((p1 + p2) / 2).astype(int))

    return mid_of_strip(base_strip), mid_of_strip(tip_strip)


def snap_to_skeleton(anchor_xy, sk_pix_rc):
    """Closest skeleton‑pixel index to anchor (x,y)."""
    diff = sk_pix_rc - np.array(anchor_xy[::-1])
    return int(np.argmin((diff ** 2).sum(axis=1)))


# -----------------------------------------------------------------------
# 2)  backbone extraction  (base → tip)
# -----------------------------------------------------------------------
def backbone_base_to_tip(filled_mask, contour):
    skel = skeletonize(filled_mask.astype(bool))
    rows, cols = np.where(skel)
    if len(rows) == 0:
        return [], None, None

    sk_pix = np.column_stack((rows, cols))            # (row,col)
    n = len(sk_pix)

    # ---- 8‑connected sparse graph -------------------------------------
    nbr = [(-1, -1), (-1, 0), (-1, 1),
           ( 0, -1),          ( 0, 1),
           ( 1, -1), ( 1, 0), ( 1, 1)]
    idx_of = {tuple(p): i for i, p in enumerate(sk_pix)}

    rr, cc = [], []
    for i, (r, c) in enumerate(sk_pix):
        for dr, dc in nbr:
            j = idx_of.get((r + dr, c + dc))
            if j is not None:
                rr.append(i)
                cc.append(j)
    G = csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n, n))

    # ---- anchors on contour & snap them --------------------------------
    base_pt, tip_pt = anchor_points_from_contour(contour)
    s_idx = snap_to_skeleton(base_pt, sk_pix)
    t_idx = snap_to_skeleton(tip_pt,  sk_pix)

    # ---- shortest path on skeleton -------------------------------------
    _, pred = dijkstra(G, indices=s_idx, return_predecessors=True)
    path_idx = []
    u = t_idx
    while u != -9999:
        path_idx.append(u)
        if u == s_idx:
            break
        u = pred[u]
    path_idx.reverse()

    # ------------------------------------------------------------------
    # 3. convert to (x,y)  +  anchor‑point enforcement
    # ------------------------------------------------------------------
    backbone = [(int(sk_pix[k, 1]), int(sk_pix[k, 0])) for k in path_idx]

    def ensure_endpoint(pt_xy, end_of_path):
        """pt_xy가 end_of_path와 멀면 앞/뒤에 삽입한다."""
        if end_of_path is None:
            return [pt_xy]
        dx = pt_xy[0] - end_of_path[0]
        dy = pt_xy[1] - end_of_path[1]
        if dx*dx + dy*dy > 25:          # ‑‑ 5 px 이상 떨어져 있으면 보정
            return [pt_xy]
        return []

    # prepend base, append tip  (거리가 5픽셀 이상이면 강제 삽입)
    backbone = ( ensure_endpoint(base_pt, backbone[0]) +
                 backbone +
                 ensure_endpoint(tip_pt,  backbone[-1]) )

    return backbone, base_pt, tip_pt


# -----------------------------------------------------------------------
# 3)  frame‑processing pipeline
# -----------------------------------------------------------------------
def process_frame(frame, roi_radius=50):
    """
    Returns
    -------
    filled_mask : binary mask of blob (uint8)
    overlay     : original frame + contour (green),
                  backbone (red), base (blue), tip (orange)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    sat = 130
    val = 130
    lower_color = np.array([40,
                            int(0 * 2.55),
                            int(0 * 2.55)], dtype=np.uint8)

    # 먼저 16‑bit로 만들고 나서 잘라낸 뒤 uint8 로 변환
    upper_color16 = np.array([170,
                            int(sat * 2.55),
                            int(val * 2.55)], dtype=np.uint16)
    upper_color = np.clip(upper_color16, 0, 255).astype(np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_color, upper_color)

    b, g, r = cv2.split(frame)
    condition = (
        (b.astype(float) > 1.1 * r.astype(float)) &
        (g.astype(float) > 1.1 * r.astype(float)) &
        (g.astype(float) > (r.astype(float) + 10.0)) &
        (b.astype(float) > (r.astype(float) + 10.0))
    )
    color_mask = np.zeros_like(r, np.uint8)
    color_mask[condition] = 255

    mask = cv2.bitwise_and(hsv_mask, color_mask)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

    # ---------- seed inside circular ROI --------------------------------
    h_img, w_img = mask.shape
    center = (w_img // 2, h_img // 2)
    roi_mask = np.zeros_like(mask)
    cv2.circle(roi_mask, center, roi_radius, 255, -1)
    seed_candidates = cv2.bitwise_and(mask, roi_mask)

    seed_point = None
    nz = cv2.findNonZero(seed_candidates)
    if nz is not None:
        seed_point = tuple(np.mean(nz, axis=0)[0].astype(int))

    flood_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
    if seed_point is not None:
        tmp = mask.copy()
        cv2.floodFill(tmp, flood_mask, seed_point, 128)
        blob_mask = (tmp == 128).astype(np.uint8) * 255
    else:
        blob_mask = mask.copy()

    # ---------- contour --------------------------------------------------
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best_contour = max(contours, key=cv2.contourArea) if contours else None

    filled_mask = np.zeros_like(blob_mask)
    if best_contour is not None:
        cv2.drawContours(filled_mask, [best_contour], -1, 255, -1)

    # clean small holes / spurs
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel3, 2)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN,  kernel3, 1)

    # ---------- backbone -------------------------------------------------
    backbone, base_pt, tip_pt = backbone_base_to_tip(filled_mask, best_contour)
    #backbone = smooth_backbone(backbone, window=7)
    backbone = spline_smooth_backbone(backbone, smoothing=1000, num=2000)


    # ---------- overlay drawing -----------------------------------------
    overlay = frame.copy()
    if best_contour is not None:
        cv2.drawContours(overlay, [best_contour], -1, (0, 255, 0), 2)

    for p, q in zip(backbone[:-1], backbone[1:]):
        cv2.line(overlay, p, q, (0, 0, 255), 2)

    if base_pt is not None:
        cv2.circle(overlay, base_pt, 4, (255, 0, 0), -1)      # blue
    if tip_pt is not None:
        cv2.circle(overlay, tip_pt, 4, (0, 128, 255), -1)     # orange

    return filled_mask, overlay


# -----------------------------------------------------------------------
# 4)  main loop
# -----------------------------------------------------------------------
cap = cv2.VideoCapture("trial1_4-22-25.avi")        # 영상 파일
# cap = cv2.VideoCapture(0)                         # 웹캠 테스트

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob_mask, overlay = process_frame(frame, roi_radius=50)

    disp_scale = 0.5
    cv2.imshow("Blob Mask",   cv2.resize(blob_mask, (0, 0), fx=disp_scale, fy=disp_scale))
    cv2.imshow("Overlay",     cv2.resize(overlay,   (0, 0), fx=disp_scale, fy=disp_scale))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
