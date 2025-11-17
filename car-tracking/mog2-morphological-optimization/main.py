import glob
import cv2
import numpy as np
# get all videos in `data/`, then extract uuid (part before.mp4). Then get associated image: data/{uuid}_last_frame.png. Then associated mask in data/masks/{uuid}_mask.png. Then filter for only those where all three files exist AND the mask contains some amount of rgb-white pixels (mask is actually labeled)

video_files = glob.glob("data/*.mp4")
image_files = glob.glob("data/*_last_frame.png")
mask_files = glob.glob("data/masks/*_mask.png")
video_ids = set([vf.split("/")[-1].split(".mp4")[0] for vf in video_files])
image_ids = set([imf.split("/")[-1].split("_last_frame.png")[0] for imf in image_files])
mask_ids = set([mf.split("/")[-1].split("_mask.png")[0] for mf in mask_files])
valid_ids = video_ids & image_ids & mask_ids
final_ids = []
for vid in valid_ids:
    mask_path = f"data/masks/{vid}_mask.png"
    mask = cv2.imread(mask_path)

    # check for >30 white pixels
    if np.sum(np.all(mask == [255, 255, 255], axis=-1)) > 30:
        final_ids.append(vid)

print("Valid IDs with non-empty masks:", final_ids)

def run_mog2(id, erode_amount, dilate_amount, iterations, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size):
    video_path = f"data/{id}.mp4"
    cap = cv2.VideoCapture(video_path)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,     
        varThreshold=10,
        detectShadows=False
    )

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = gray
        img = cv2.GaussianBlur(img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        mask = fgbg.apply(img)

        # Apply morphological operations
        for _ in range(iterations):
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)

    cap.release()

    final_mask = mask
    true_mask = cv2.imread(f"data/masks/{id}_mask.png", cv2.IMREAD_GRAYSCALE)
    true_mask = true_mask == 255
    final_mask = final_mask == 255

    mse = np.mean((final_mask.astype(float) - true_mask.astype(float)) ** 2)
    return mse
