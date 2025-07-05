#!/usr/bin/env python3
"""
3‑DoF SBA 백본‑센터라인 추적  –  색상 기반 마스크 버전
"""

import cv2, numpy as np, pandas as pd
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops

# ---------- 사용자 조정 파라미터 ----------
XZ_VIDEO   = "xz_trial_ex_4-22-25.avi"
YZ_VIDEO   = "yz_trial_ex_4-22-25.avi"
MIN_AREA   = 1_000          # 잡음 제거용 최소 픽셀 수
OUT_CSV    = "centerline_xyz.csv"
VISUALIZE  = True
# --- 색상 한계값 (필름 색상에 맞춰 필요 시 조정) ---
LOWER_HSV  = np.array([40,  0,   0   ], dtype=np.uint8)
UPPER_HSV  = np.array([170, 130, 130 ], dtype=np.uint8)
B_OVER_R   = 1.1            # b > 1.1*r  &  g > 1.1*r
DELTA_BG   = 10.0           # b,g − r  ≥ 10
# -----------------------------------------------
# ─── 새 임계값 ─────────────────────────────
H_MIN, H_MAX = 68, 95        # Hue 68‑95
S_MIN        = 80            # Sat ≥ 80
V_MIN        = 20
A_MAX        = 112           # Lab‑a ≤ 112  (녹색대)
MIN_PIXELS   = 1_000
# ─────────────────────────────────────────

from scipy.ndimage import convolve

PRUNE_LEN = 20   # 픽셀

def thin(img_bin):
    return cv2.ximgproc.thinning(img_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

def prune_skeleton(skel):
    # 8‑connected endpoint 커널
    k = np.array([[1,1,1],
                  [1,10,1],
                  [1,1,1]], np.uint8)

    while True:
        # 끝점 = 합계 11 (자기 10 + 이웃 1)
        endpoints = (convolve(skel//255, k, mode='constant') == 11)
        coords = np.column_stack(np.where(endpoints))
        if coords.size == 0:
            break

        removed_any = False
        for y,x in coords:
            # 각 끝점부터 BFS 로 브랜치 길이 측정
            stack, visited = [(y,x)], set([(y,x)])
            path = []
            while stack and len(path) < PRUNE_LEN:
                cy,cx = stack.pop()
                path.append((cy,cx))
                for ny in range(cy-1, cy+2):
                    for nx in range(cx-1, cx+2):
                        if (ny,nx) in visited: continue
                        if skel[ny,nx]:
                            stack.append((ny,nx)); visited.add((ny,nx))
            if len(path) < PRUNE_LEN:
                for py,px in path:
                    skel[py,px] = 0
                removed_any = True
        if not removed_any:
            break
    return skel

def segment(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mask_hsv = cv2.inRange(hsv, (H_MIN, S_MIN, V_MIN), (H_MAX, 255, 255))
    mask_lab = cv2.inRange(lab, (0, 0, 0),          (255, A_MAX, 255))

    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    # 모폴로지 & 가장 큰 CC만
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    labels = label(mask > 0)
    if labels.max():
        largest = max(regionprops(labels), key=lambda p:p.area)
        mask = (labels == largest.label).astype(np.uint8)*255
    return mask

def get_skeleton(mask):
    return (skeletonize(mask // 255) * 255).astype(np.uint8)


def extract_centerline_pts(skel):
    pts = np.column_stack(np.where(skel > 0))      # (row=z, col=x|y)
    return pts[np.argsort(pts[:, 0])] if pts.size else None


def match_centerlines(cl_xz, cl_yz):
    if cl_xz is None or cl_yz is None:
        return None
    z_vals = np.intersect1d(cl_xz[:, 0], cl_yz[:, 0])
    xyz = [
        [cl_xz[cl_xz[:, 0] == z][0, 1],
         cl_yz[cl_yz[:, 0] == z][0, 1],
         z]
        for z in z_vals
    ]
    return np.array(xyz) if xyz else None


def main():
    if not (Path(XZ_VIDEO).exists() and Path(YZ_VIDEO).exists()):
        raise FileNotFoundError("동영상 파일을 같은 폴더에 두었는지 확인하세요.")

    cap_xz = cv2.VideoCapture(XZ_VIDEO)
    cap_yz = cv2.VideoCapture(YZ_VIDEO)

    records, frame_idx = [], 0
    while True:
        ret1, f1 = cap_xz.read()
        ret2, f2 = cap_yz.read()
        if not (ret1 and ret2):
            break

        mask1, mask2 = segment(f1), segment(f2)
        cv2.imshow("mask1", mask1)
        cv2.imshow("mask2", mask2)
        skel1, skel2 = prune_skeleton(thin((mask1))), prune_skeleton(thin((mask2)))
        cl1,  cl2   = extract_centerline_pts(skel1), extract_centerline_pts(skel2)

        cl3d = match_centerlines(cl1, cl2)
        if cl3d is not None:
            records.extend([[frame_idx, *pt] for pt in cl3d])

        if VISUALIZE:
            disp1 = cv2.addWeighted(f1, 0.5, cv2.cvtColor(skel1, cv2.COLOR_GRAY2BGR), 0.5, 0)
            disp2 = cv2.addWeighted(f2, 0.5, cv2.cvtColor(skel2, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow("xz view + skeleton", disp1)
            cv2.imshow("yz view + skeleton", disp2)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap_xz.release(), cap_yz.release(), cv2.destroyAllWindows()

    if records:
        pd.DataFrame(records, columns=["frame", "x_px", "y_px", "z_px"]).to_csv(OUT_CSV, index=False)
        print(f"[✓] {len(records)} points saved to {OUT_CSV}")
    else:
        print("[!] 추출된 중심선 포인트가 없습니다.")


if __name__ == "__main__":
    main()
