import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def anchor_points_from_contour(cnt, strip=0.07):
    """
    Parameters
    ----------
    cnt   : Nx1x2 array returned by cv2.findContours
    strip : fraction of bbox size that will be treated as 'edge strip'
            e.g. 0.07 → lowest 7 % of the blob height is the base strip,
                        right‑most 7 % of the blob width is the tip strip
    Returns
    -------
    base_pt (x,y), tip_pt (x,y)
    """
    pts = cnt.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    h, w = (y_max - y_min), (x_max - x_min)
    base_strip = pts[pts[:,1] >= y_max - strip*h]        # lowest strip
    tip_strip  = pts[pts[:,0] >= x_max - strip*w]        # right‑most strip

    # --- robust line fit with PCA (gives a direction vector and a centroid) ---
    def mid_of_strip(strip_pts):
        if len(strip_pts) < 2:                     # fall‑back
            return tuple(strip_pts[0])
        mu = strip_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(strip_pts - mu)
        dir_vec = vt[0]                            # principal component
        # project points onto line, take extremes → segment endpoints
        t = (strip_pts - mu) @ dir_vec
        p1, p2 = mu + dir_vec*t.min(), mu + dir_vec*t.max()
        return tuple(((p1+p2)/2).astype(int))      # midpoint of segment

    base_pt = mid_of_strip(base_strip)
    tip_pt  = mid_of_strip(tip_strip)
    return base_pt, tip_pt

def snap_to_skeleton(anchor, skel_pix):
    """Return index of skeleton pixel that is closest to `anchor` (x,y)."""
    diff = skel_pix - np.array(anchor[::-1])        # (row,col) vs (y,x)
    idx  = np.argmin((diff**2).sum(axis=1))
    return idx

def backbone_base_to_tip(filled_mask, contour):
    skel = skeletonize(filled_mask.astype(bool))
    rows, cols = np.where(skel)
    if len(rows) == 0:
        return []

    sk_pix   = np.column_stack((rows, cols))              # (n,2)
    idx_of   = {tuple(p): i for i, p in enumerate(sk_pix)}

    # build 8‑connected sparse graph (same as before) …
    #   ---> returns adjacency matrix `G`

    # 1. anchors ---------------------------------------------------------------
    base_pt, tip_pt = anchor_points_from_contour(contour)
    s_idx = snap_to_skeleton(base_pt, sk_pix)
    t_idx = snap_to_skeleton(tip_pt , sk_pix)

    # 2. shortest path on the skeleton graph ----------------------------------
    _, pred = dijkstra(G, indices=s_idx, return_predecessors=True)
    path_idx = []
    u = t_idx
    while u != -9999:
        path_idx.append(u)
        if u == s_idx: break
        u = pred[u]
    path_idx = path_idx[::-1]

    # 3. convert to (x,y) ------------------------------------------------------
    backbone = [(int(sk_pix[k,1]), int(sk_pix[k,0])) for k in path_idx]
    return backbone, base_pt, tip_pt

def get_backbone(blob_mask):
    """Return ordered centre‑line pixels (x,y) from base to tip."""
    skel = skeletonize(blob_mask.astype(bool))
    if not skel.any():
        return []

    # ---- build list of skeleton pixels and a mapping idx <-> (row,col)
    rows, cols = np.where(skel)
    pix      = list(zip(rows, cols))               # [(r,c), ...]
    idx_of   = {p: i for i, p in enumerate(pix)}   # (r,c) -> index

    # ---- neighbourhood offsets (8‑connectivity)
    nbrs = [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),         ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)]

    # ---- build adjacency list
    n = len(pix)
    rows_idx, cols_idx = [], []
    for i, (r, c) in enumerate(pix):
        for dr, dc in nbrs:
            q = (r+dr, c+dc)
            if q in idx_of:
                j = idx_of[q]
                rows_idx.append(i)
                cols_idx.append(j)
    data = np.ones(len(rows_idx), dtype=np.float32)
    G = csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))

    # ---- find endpoints: degree == 1
    deg = np.array(G.sum(axis=1)).ravel()
    endpoints = np.where(deg == 1)[0]
    if len(endpoints) < 2:
        return [tuple(reversed(p)) for p in pix]   # already a line

    # all‑pairs distances (row = endpoint, col = *all* nodes)
    dist_full = dijkstra(G, directed=False, indices=endpoints)

    # keep only columns that are also endpoints  →  L × L matrix
    dist_ep = dist_full[:, endpoints]

    # ignore unreachable pairs
    dist_ep[~np.isfinite(dist_ep)] = -1

    # longest finite path between two endpoints
    i_best, j_best = np.unravel_index(dist_ep.argmax(), dist_ep.shape)
    s_idx = endpoints[i_best]       # start
    t_idx = endpoints[j_best]       # target  (now guaranteed to be an endpoint)

    # ---- retrieve shortest path between s_idx and t_idx
    _, predecessors = dijkstra(G, directed=False, indices=s_idx, return_predecessors=True)
    path_idx = []
    u = t_idx
    while u != -9999:          # ‑9999 = source sentinel
        path_idx.append(u)
        if u == s_idx:
            break
        u = predecessors[u]
    path_idx = path_idx[::-1]  # from s -> t

    # ---- convert to (x,y) pairs (col,row) for drawing
    backbone = [(pix[k][1], pix[k][0]) for k in path_idx]
    return backbone

def process_frame(frame, roi_radius=50, method='skeleton'):
    """
    Process a single frame:
      1) Rotate 180 degrees.
      2) Create a mask of 'greenish' pixels using both HSV and BGR conditions.
      3) Use a circular ROI centered in the frame to select a seed point.
      4) Flood-fill from the seed to extract the connected blob.
      5) Extract the blob's contour.
      6) Compute the backbone using the specified method ('skeleton' or 'dp').
      7) Return the blob mask and an overlay image with the contour and backbone.
    """
    # (A) Rotate frame 180 degrees.
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # (B) Convert to HSV and threshold.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_color = np.array([50, 100, 100])
    #upper_color = np.array([70, 255, 255])
    lower_color = np.array([40, 0*2.55, 0*2.55]) 
    upper_color = np.array([170, 130*2.55, 130*2.55])
    hsv_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # (C) BGR-based mask.
    b, g, r = cv2.split(frame)
    condition = (
        (b.astype(float) > 1.1 * r.astype(float)) &
        (g.astype(float) > 1.1 * r.astype(float)) &
        (g.astype(float) > (r.astype(float) + 10.0)) &
        (b.astype(float) > (r.astype(float) + 10.0)) #&
        #(np.abs(b.astype(float) - g.astype(float)) < 15.0)
    )
    color_mask = np.zeros_like(r, dtype=np.uint8)
    color_mask[condition] = 255
    
    # (D) Combined mask.
    mask = cv2.bitwise_and(hsv_mask, color_mask)
    
    # (E) Morphological filtering.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # (F) Create circular ROI for seed selection.
    h_img, w_img = mask.shape
    center = (w_img // 2, h_img // 2)
    roi_mask = np.zeros_like(mask)
    cv2.circle(roi_mask, center, roi_radius, 255, thickness=-1)
    seed_candidates = cv2.bitwise_and(mask, roi_mask)
    
    # (G) Choose a seed point.
    seed_point = None
    non_zero = cv2.findNonZero(seed_candidates)
    if non_zero is not None:
        seed_point = tuple(np.mean(non_zero, axis=0)[0].astype(int))
    
    # (H) Flood-fill to extract the blob.
    blob_mask = np.zeros((h_img+2, w_img+2), np.uint8)
    if seed_point is not None:
        flood_mask = mask.copy()
        cv2.floodFill(flood_mask, blob_mask, seed_point, 128)
        blob_mask = np.uint8(flood_mask == 128) * 255
    else:
        blob_mask = mask.copy()
    
    # (I) Find the largest contour.
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_contour = cnt
    
    # (J) Create a filled blob mask from the best contour.
    filled_mask = np.zeros_like(blob_mask)
    if best_contour is not None:
        cv2.drawContours(filled_mask, [best_contour], -1, 255, thickness=-1)

    # (J‑bis) ------- MORPHOLOGICAL CLEAN‑UP (step 2) -------
    # close small holes and remove spurs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN , kernel, iterations=1)


    # (K) Compute the backbone.
    backbone_points = get_backbone(filled_mask)

    # (L) Create overlay.
    overlay = frame.copy()
    if best_contour is not None:
        cv2.drawContours(overlay, [best_contour], -1, (0, 255, 0), 2)
    #for i in range(1, len(backbone_points)):
        #cv2.line(overlay, backbone_points[i-1], backbone_points[i], (0, 0, 255), 2)
    
    return filled_mask, overlay

# Main processing loop.
cap = cv2.VideoCapture("trial1_4-22-25.avi")
#cap = cv2.VideoCapture(0)

cnt = 0
while True:
    ret, frame = cap.read() 

    if not ret:
        break
    
    cnt = cnt + 1
    if cnt > 55:
        blob_mask, overlay = process_frame(frame, roi_radius=50, method='dp')
        
        # Resize images to half the original dimensions.
        display_scale = 0.5
        resized_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
        resized_mask = cv2.resize(blob_mask, (0, 0), fx=display_scale, fy=display_scale)
        resized_overlay = cv2.resize(overlay, (0, 0), fx=display_scale, fy=display_scale)
        
        cv2.imshow("Rotated Frame", resized_frame)
        cv2.imshow("Blob Mask", resized_mask)
        cv2.imshow("Overlay (Contour & Backbone)", resized_overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

cap.release()
cv2.destroyAllWindows()
