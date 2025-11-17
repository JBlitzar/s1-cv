import glob
import cv2
import numpy as np
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
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

def run_mog2(id, erode_amount, dilate_amount, gaussian_blur_kernel_size, erode_kernel_size, dilate_kernel_size, history, var_threshold, erode_before_dilate=False, alpha=1.5, beta=0):
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

    cap.release()

    final_mask = mask
    true_mask = cv2.imread(f"data/masks/{id}_mask.png", cv2.IMREAD_GRAYSCALE)
    true_mask = true_mask == 255
    final_mask = final_mask == 255

    mse = np.mean((final_mask.astype(float) - true_mask.astype(float)) ** 2)
    return mse

def objective(trial):
    erode_amount = trial.suggest_int('erode_amount', 1, 5)
    dilate_amount = trial.suggest_int('dilate_amount', 1, 5)
    gaussian_blur_kernel_size = trial.suggest_categorical('gaussian_blur_kernel_size', [3, 5, 7, 9, 11,13,15,17,19,25,45])
    erode_kernel_size = trial.suggest_int('erode_kernel_size', 1, 15, step=2)
    dilate_kernel_size = trial.suggest_int('dilate_kernel_size', 1, 15, step=2)
    history = trial.suggest_int('history', 100, 1000)
    var_threshold = trial.suggest_float('var_threshold', 2.0, 50.0)
    erode_before_dilate = trial.suggest_categorical('erode_before_dilate', [True, False])
    alpha = trial.suggest_float('alpha', 0.25, 3.0)
    beta = trial.suggest_float('beta', -50, 50)
    
    total_mse = 0.0
    valid_count = 0
    
    for video_id in tqdm(final_ids, leave=False):
        try:
            mse = run_mog2(
                video_id, 
                erode_amount, 
                dilate_amount, 
                gaussian_blur_kernel_size, 
                erode_kernel_size, 
                dilate_kernel_size,
                history,
                var_threshold,
                erode_before_dilate,
                alpha,
                beta
            )
            total_mse += mse
            valid_count += 1
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue
    
    if valid_count == 0:
        return float('inf')
    
    average_mse = total_mse / valid_count
    print(f"Trial {trial.number}: Average MSE = {average_mse:.6f}")
    return average_mse

if __name__ == "__main__":
    print(f"Starting optimization with {len(final_ids)} valid videos")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=200, show_progress_bar=True, n_jobs=4)
    
    print("\nOptimization completed!")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best MSE: {study.best_value:.6f}")
    
    with open('best.txt', 'w') as f:
        f.write("Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Best MSE: {study.best_value:.6f}\n")
    
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv('optuna_results.csv', index=False)
