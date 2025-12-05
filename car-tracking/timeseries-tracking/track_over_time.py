import random
from run_optuna import final_ids
import cv2
import numpy as np
from mog2_pipeline import run_mog2
import subprocess
from dataclasses import dataclass, field


def load_params():
    with open("best.txt", "r") as f:
        lines = f.readlines()
        params = {}
        for line in lines:
            if not line.startswith("  "):
                continue
            key, value = line.strip().split(": ")
            try:
                params[key] = float(value)
            except ValueError:
                if value == "True":
                    params[key] = True
                elif value == "False":
                    params[key] = False
                else:
                    params[key] = value
        return params


global blobList
blobList = []


@dataclass
class Blob:
    x: float
    y: float
    radius: float
    dx: float = 0.0
    dy: float = 0.0
    distances: list = field(default_factory=list)
    radii: list = field(default_factory=list)


SEARCH_PADDING = 2
MIN_RADIUS = 5.0


def _boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _intersection_blob(label_id, labels, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(labels.shape[1], x2)), int(min(labels.shape[0], y2))
    roi = labels[y1:y2, x1:x2]
    mask = roi == label_id
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    cx = x1 + xs.mean()
    cy = y1 + ys.mean()
    radius = np.sqrt(mask.sum() / np.pi)
    return cx, cy, radius


def tracking_callback(frame, gray, mask, img, bg, L_ratio):
    frame = frame.copy()
    global blobList
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    detections = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        radius = min(w, h) / 2
        if radius < MIN_RADIUS:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        detections.append(
            {
                "label": i,
                "bbox": (x, y, x + w, y + h),
                "matched": False,
                "blob": Blob(cx, cy, radius),
            }
        )

    
    mask_colored = cv2.applyColorMap(mask.astype(np.uint8) * 255, cv2.COLORMAP_BONE)
    frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    blobList = [blob for blob in blobList if blob.radius >= MIN_RADIUS]

    if not blobList:
        blobList = []
        for det in detections:
            det_blob = det["blob"]
            det_blob.radii = [det_blob.radius]
            blobList.append(det_blob)

    updated = []
    for blob in blobList:
        pred_x = blob.x + blob.dx
        pred_y = blob.y + blob.dy
        pred_bbox = (
            pred_x - blob.radius - SEARCH_PADDING,
            pred_y - blob.radius - SEARCH_PADDING,
            pred_x + blob.radius + SEARCH_PADDING,
            pred_y + blob.radius + SEARCH_PADDING,
        )
        candidates = [
            idx
            for idx, det in enumerate(detections)
            if (not det["matched"]) and _boxes_intersect(pred_bbox, det["bbox"])
        ]

        for idx in candidates:
            det = detections[idx]
            intersection = _intersection_blob(det["label"], labels, pred_bbox)
            if intersection is None:
                continue
            nx, ny, nr = intersection
            if nr < MIN_RADIUS:
                continue
            det["matched"] = True
            new_blob = Blob(nx, ny, nr)
            new_blob.dx = nx - blob.x
            new_blob.dy = ny - blob.y
            new_blob.distances = blob.distances.copy()
            new_blob.distances.append(np.hypot(new_blob.dx, new_blob.dy))
            new_blob.radii = blob.radii.copy()
            new_blob.radii.append(nr)
            updated.append(new_blob)


       

    for det in detections:
        if det["matched"]:
            continue
        det_blob = det["blob"]
        det_blob.radii = [det_blob.radius]
        updated.append(det_blob)

    blobList = updated
    
    for blob in blobList:
        start_point = (int(blob.x - blob.dx), int(blob.y - blob.dy))
        end_point = (int(blob.x), int(blob.y))
        
        avg_distance = sum(blob.distances) / len(blob.distances) if blob.distances else 0.0
        avg_radius = sum(blob.radii) / len(blob.radii) if blob.radii else 0.0
        if avg_distance > 0.0 and avg_radius > 0.0 and blob.dx != 0.0 and blob.dy != 0.0:
            cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"D: {avg_distance:.1f}, R: {avg_radius:.1f}",
                (int(blob.x + 10), int(blob.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
            # print(blob)

    cv2.imshow("Tracking", frame)
    cv2.waitKey(1000 // 10)


if __name__ == "__main__":
    params = load_params()
    # video_id = random.choice(final_ids)
    run_mog2(
        # 0,
        "5bd8fa3e-2ed9-40f7-b47a-04e65210a9f3",
        int(params["erode_amount"]),
        int(params["dilate_amount"]),
        int(params["gaussian_blur_kernel_size"]),
        int(params["erode_kernel_size"]),
        int(params["dilate_kernel_size"]),
        int(params["history"]),
        float(params["var_threshold"]),
        bool(params["erode_before_dilate"]),
        float(params["L_ratio_thresh"]),
        callback=tracking_callback,
    )
