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

    distances: list = field(default_factory=list)
    radii: list = field(default_factory=list)


def tracking_callback(frame, gray, mask, img, bg, L_ratio):
    frame = frame.copy()
    global blobList
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    _blobs = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        _blobs.append(Blob(cx, cy, min(w, h) / 2, distances=[]))
    if len(blobList) == 0:
        blobList = _blobs
        for blob in blobList:
            blob.radii = [blob.radius]
    if len(blobList) > 0:
        newBlobs = []
        for blob in _blobs:
            closest_blob = min(
                blobList,
                key=lambda b: (b.x - blob.x) ** 2 + (b.y - blob.y) ** 2,
            )
            cv2.line(
                frame,
                (int(blob.x), int(blob.y)),
                (int(closest_blob.x), int(closest_blob.y)),
                (255, 0, 0),
                2,
            )

            distance = np.sqrt(
                (blob.x - closest_blob.x) ** 2 + (blob.y - closest_blob.y) ** 2
            )
            blob.distances.extend(closest_blob.distances)
            blob.distances.append(distance)

            blob.radii.extend(closest_blob.radii)
            blob.radii.append(blob.radius)

            newBlobs.append(blob)
        blobList = newBlobs
        # print(blobList)

        for blob in blobList:
            avg_distance = sum(blob.distances) / len(blob.distances)
            avg_radius = sum(blob.radii) / len(blob.radii)
            cv2.putText(
                frame,
                f"D: {avg_distance:.1f}, R: {avg_radius:.1f}",
                (int(blob.x + 10), int(blob.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

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
