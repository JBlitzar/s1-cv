import random
from run_optuna import final_ids
import cv2
import numpy as np


def run_mog2_stream(id, erode_amount, dilate_amount, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size, history, var_threshold, erode_before_dilate=False, clipLimit=1.5, tileGridSize=8):
    video_path = f"data/{id}.mp4"
    cap = cv2.VideoCapture(video_path)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,     
        varThreshold=var_threshold,
        detectShadows=True
    )


    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame
        img = cv2.GaussianBlur(img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

        mask = fgbg.apply(img)

        cv2.imshow("mog2", mask)

        if erode_before_dilate:
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
        else:
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)

        cv2.imshow("Raw Frame", frame)
        cv2.imshow("Processed Frame", img)
        cv2.imshow("Motion Mask", mask)
        
        alpha = 0.5  # transparency level (0 = invisible, 1 = fully red)

        superimposed = img.copy().astype(np.float32)
        superimposed /= np.max(superimposed) + 1e-10
        superimposed *= 255.0
        superimposed = superimposed.astype(np.uint8)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_GRAY2BGR)

        # Create red overlay
        red_overlay = np.zeros_like(superimposed)
        red_overlay[:] = (0, 0, 255)  # red in BGR

        # Blend only where mask == 255
        blend_region = (mask == 255)
        superimposed[blend_region] = (
            alpha * red_overlay[blend_region] +
            (1 - alpha) * superimposed[blend_region]
        ).astype(np.uint8)

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
                float(params['clipLimit']),
                int(params['tileGridSize'])
            )