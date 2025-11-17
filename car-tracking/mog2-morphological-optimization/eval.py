import random
from run_optuna import final_ids
import cv2
import numpy as np


def run_mog2_stream(id, erode_amount, dilate_amount, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size, history, var_threshold, erode_before_dilate=False, alpha=1.5, beta=0):
    video_path = f"data/{id}.mp4"
    cap = cv2.VideoCapture(video_path)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,     
        varThreshold=var_threshold,
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
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        mask = fgbg.apply(img)

        if erode_before_dilate:
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
        else:
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)

        cv2.imshow("Raw Frame", frame)
        cv2.imshow("Processed Frame", img)
        cv2.imshow("Motion Mask", mask)
        superimposed = img.copy()
        superimposed = np.float32(superimposed)
        superimposed /= np.max(superimposed) + 1e-10
        superimposed *= 255.0
        superimposed = np.uint8(superimposed)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_GRAY2BGR)
        superimposed[mask == 255] = [0, 0, 255]
        cv2.imshow("Superimposed", superimposed)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break   

    i = input()
    cap.release()



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
    
if __name__ == "__main__":
    params = load_params()
    video_id = random.choice(final_ids)
    run_mog2_stream(
                video_id, 
                int(params['erode_amount']), 
                int(params['dilate_amount']), 
                int(params['gaussian_blur_kernel_size']), 
                int(params['erode_kernel_size']), 
                int(params['dilate_kernel_size']),
                int(params['history']),
                float(params['var_threshold']),
                bool(params['erode_before_dilate']),
                float(params['alpha']),
                float(params['beta'])
            )