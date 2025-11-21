import queue
import cv2
import numpy as np
from get_videos import urls
import subprocess


def _get_stream_info(url):
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
    
    return 1920, 1080 

def run_mog2(id, erode_amount, dilate_amount, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size, history, var_threshold, erode_before_dilate=False, L_ratio_thresh=1.1, callback=None):
    # print("ARGS: ")
    # print(f"id: {id}")
    # print(f"erode_amount: {erode_amount}")
    # print(f"dilate_amount: {dilate_amount}")
    # print(f"gaussian_blur_kernel_size: {gaussian_blur_kernel_size}")
    # print(f"erode_kernel_size: {erode_kernel_size}")
    # print(f"dilate_kernel_size: {dilate_kernel_size}")
    # print(f"history: {history}")
    # print(f"var_threshold: {var_threshold}")
    # print(f"erode_before_dilate: {erode_before_dilate}")
    # print(f"L_ratio_thresh: {L_ratio_thresh}")
    # print(f"callback: {callback}")
    video_path = ""

    if isinstance(id, int) and 0 <= id < len(urls):
        url = urls[id]
        

        width, height = _get_stream_info(url)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
            '-'
        ]
        

        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        is_stream = True
    else:

        video_path = f"data/{id}.mp4"
        cap = cv2.VideoCapture(video_path)
        is_stream = False
        #print("Processing video:", video_path)



    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,     
        varThreshold=var_threshold,
        detectShadows=True # detectShadows = true makes it not detect shadows. 
    )


    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask = None
    while True:
        if is_stream:

            raw_frame = ffmpeg_process.stdout.read(width * height * 3)
            if len(raw_frame) != width * height * 3:
                break
            
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = gray
        img = frame
        img = cv2.GaussianBlur(img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

        mask = fgbg.apply(img)

        bg = fgbg.getBackgroundImage()

        mask = mask > 200
        mask = mask.astype(np.uint8) * 255

        lab_f = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)

        L_f = lab_f[:,:,0].astype(np.float32)
        L_b = lab_b[:,:,0].astype(np.float32)



        L_ratio = (L_f + 1) / (L_b + 1)

        a = L_ratio > L_ratio_thresh


        mask = mask.astype(np.uint8) * 255 & (a.astype(np.uint8) * 255)
        

        if erode_before_dilate:
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
        else:
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_amount)
            mask = cv2.erode(mask, erode_kernel, iterations=erode_amount)
    
        if callback is not None:
            callback(frame, gray, mask, img, bg)

    cap.release()

    
    return mask


def run_mog2_mse(id, *args, **kwargs):
    mask =  run_mog2(id, *args, **kwargs)
    final_mask = mask
    true_mask = cv2.imread(f"data/masks/{id}_mask.png", cv2.IMREAD_GRAYSCALE)
    true_mask = true_mask == 255
    #print(true_mask)


    mse = np.mean((final_mask.astype(float) - true_mask.astype(float)) ** 2)
    return mse

def _tilewindows(windows, width, height):
    n = len(windows)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    for i, win in enumerate(windows):
        x = (i % cols) * width
        y = (i // cols) * height
        cv2.moveWindow(win, x, y)

def visualization_callback(frame, gray, mask, img, bg):
    cv2.imshow("Raw Frame", frame)
    cv2.imshow("Processed Frame", img)
    cv2.imshow("Motion Mask", mask)
    superimposed = gray.copy()
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



    
    _tilewindows(["Raw Frame", "Processed Frame", "Motion Mask", "Superimposed", "Distance Transform"], 640//2, 480//2)


    

    cv2.waitKey(30)

def run_mog2_info(*args, **kwargs):
    run_mog2(*args, callback=visualization_callback, **kwargs)


    
    