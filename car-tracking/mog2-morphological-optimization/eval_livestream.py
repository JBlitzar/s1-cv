import random
from run_optuna import final_ids
import cv2
import numpy as np
from get_videos import urls
import subprocess
import threading
import sys


def run_mog2_stream(id, erode_amount, dilate_amount, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size, history, var_threshold, erode_before_dilate=False, alpha=1.5, beta=0):
    # Check if id is a valid index for urls list
    if isinstance(id, int) and 0 <= id < len(urls):
        # Stream from m3u8 URL using ffmpeg
        url = urls[id]
        
        # Get actual stream dimensions
        width, height = get_stream_info(url)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # no audio
            '-'  # output to stdout
        ]
        
        # Start ffmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        is_stream = True
    else:
        # Use local video file
        video_path = f"data/{id}.mp4"
        cap = cv2.VideoCapture(video_path)
        is_stream = False

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,     
        varThreshold=var_threshold,
        detectShadows=False
    )

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    erode_kernel_v2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        if is_stream:
            # Read frame from ffmpeg stdout
            raw_frame = ffmpeg_process.stdout.read(width * height * 3)
            if len(raw_frame) != width * height * 3:
                break
            
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            ret = True
        else:
            # Read from local video file
            ret, frame = cap.read()
            if not ret:
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = gray
        img = cv2.GaussianBlur(img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        mask = fgbg.apply(img)

        cv2.imshow("mog2", mask)

        if erode_before_dilate:
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
        else:
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)

        # mask = cv2.erode(mask, erode_kernel_v2, iterations=1)

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

        # Apply distance transform to the mask
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_transform_colored = cv2.applyColorMap(dist_transform_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Distance Transform", dist_transform_colored)
        
        

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break   

    if is_stream:
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
    else:
        cap.release()

    cv2.destroyAllWindows()


def get_stream_info(url):
    """Get stream dimensions using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        import json
        data = json.loads(result.stdout)
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return stream['width'], stream['height']
    except:
        pass
    
    return 1920, 1080  # Default fallback


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
    
    # Use index 0 from urls list instead of urls[0] directly
    run_mog2_stream(
                0,  # This will use urls[0]
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