import random
from run_optuna import final_ids
import cv2
import numpy as np
from mog2_pipeline import run_mog2_info
import subprocess


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
    run_mog2_info(video_id,int(params['erode_amount']), 
                int(params['dilate_amount']), 
                int(params['gaussian_blur_kernel_size']), 
                int(params['erode_kernel_size']), 
                int(params['dilate_kernel_size']),
                int(params['history']),
                float(params['var_threshold']),
                bool(params['erode_before_dilate']),
                float(params['L_ratio_thresh']),
                )