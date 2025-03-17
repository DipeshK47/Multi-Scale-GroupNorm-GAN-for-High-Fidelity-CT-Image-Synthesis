import glob
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp

NUM_JOBS = 8
IMG_SIZE = 256

# Update INPUT_DATA_DIR to be the directory containing your .nii files.
INPUT_DATA_DIR = '/Users/dipeshkumar/Desktop/Major Project/code/GSP/Sub1570_S1/T1_MEMPRAGE_RMS/2014-06-30_00_00_00.0/I464942'
OUTPUT_DATA_DIR = '/Users/dipeshkumar/Desktop/Major Project/code/preprocess_op'

LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
# Change SUFFIX to '.nii' if your files have that extension.
SUFFIX = '.nii'
TRIM_BLANK_SLICES = True

def resize_img(img):
    # Replace NaNs with LOW_THRESHOLD
    nan_mask = np.isnan(img)
    img[nan_mask] = LOW_THRESHOLD

    # Normalize image intensity between -1 and 1
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1, 1])

    if TRIM_BLANK_SLICES:
        # Remove slices where the mean value equals -1 (blank slices)
        valid_plane_i = np.mean(img, axis=(1, 2)) != -1
        img = img[valid_plane_i, :, :]

    # Resize image to a cube of dimensions IMG_SIZE^3
    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img

def batch_resize(batch_idx, img_list):
    for idx in range(len(img_list)):
        if idx % NUM_JOBS != batch_idx:
            continue
        img_path = img_list[idx]
        imgname = os.path.basename(img_path)
        # Construct output file name using os.path.join for better path handling
        output_file = os.path.join(OUTPUT_DATA_DIR, os.path.splitext(imgname)[0] + ".npy")

        # Skip file if it already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            continue
        
        try:
            # Read the image from the full path directly
            img = sitk.ReadImage(img_path)
        except Exception as e:
            print(f"Image loading error: {imgname}, error: {e}")
            continue 

        img = sitk.GetArrayFromImage(img)
        
        try:
            img = resize_img(img)
        except Exception as e:
            print(f"Image resize error: {imgname}, error: {e}")
            continue

        try:
            np.save(output_file, img)
            print(f"Success: Processed {imgname} saved at {output_file}")
        except Exception as e:
            print(f"File saving error: {imgname}, error: {e}")

def main():
    try:
        # Find all images with the given SUFFIX in the INPUT_DATA_DIR
        img_list = list(glob.glob(os.path.join(INPUT_DATA_DIR, "*" + SUFFIX)))
        if not img_list:
            print("Error: No images found in the input directory.")
            return
        
        processes = []
        for i in range(NUM_JOBS):
            p = mp.Process(target=batch_resize, args=(i, img_list))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        print(f"Success: All images processed. Output files are saved in: {OUTPUT_DATA_DIR}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == '__main__':
    main()
