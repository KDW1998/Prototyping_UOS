'''
python tools/overlay_and_save_gt.py --img_dir '/home/deogwonkang/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/합성이미지/BR/원본이미지/br_CutMix/100units/Random/br_19111_01_to_br_19111_104/leftImg8bit/train' --gt_dir '/home/deogwonkang/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/합성이미지/BR/원본이미지/br_CutMix/100units/Random/br_19111_01_to_br_19111_104/DEPT_gtFine/train' --output_dir '/home/deogwonkang/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/합성이미지/BR/원본이미지/br_CutMix/100units/Random/br_19111_01_to_br_19111_104/visualization'
'''
import argparse
import os
import numpy as np
from glob import glob
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Extended color mapping
color_mapping = {
    0: [0, 0, 0],        # Class 0 in black
    1: [0, 0, 255],      # Class 1 in Red
    2: [0, 255, 255],    # Class 2 in Yellow
    3: [255, 0, 0],      # Class 3 in Blue
    4: [0, 255, 0],      # Class 4 in Green
    5: [255, 0, 255],    # Class 5 in Magenta
    6: [255, 255, 0],    # Class 6 in Cyan
    7: [128, 0, 0],      # Class 7 in Maroon
    8: [0, 128, 0]       # Class 8 in Dark Green
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize images and ground truths')
    parser.add_argument('--img_dir', required=True, help='the directory containing original images')
    parser.add_argument('--gt_dir', required=True, help='the directory containing ground truth files')
    parser.add_argument('--output_dir', required=True, help='the directory to save the visualized ground truth files')
    parser.add_argument('--target_label', type=int, default=1, help='the label to filter images (default: 1)')
    args = parser.parse_args()
    return args.img_dir, args.gt_dir, args.output_dir, args.target_label

def get_matching_files(img_dir, gt_dir):
    img_files = glob(f'{img_dir}/*_leftImg8bit.png')
    gt_files = glob(f'{gt_dir}/*_gtFine_labelIds.png')
    
    img_dict = {os.path.basename(f).replace('_leftImg8bit.png', ''): f for f in img_files}
    gt_dict = {os.path.basename(f).replace('_gtFine_labelIds.png', ''): f for f in gt_files}
    
    common_keys = set(img_dict.keys()) & set(gt_dict.keys())
    
    matched_pairs = [(img_dict[k], gt_dict[k]) for k in common_keys]
    return matched_pairs

def main(img_dir, gt_dir, output_dir, target_label):
    file_pairs = get_matching_files(img_dir, gt_dir)
    print(f"Found {len(file_pairs)} matching pairs of image and ground truth files")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for img_file, gt_file in file_pairs:
        try:
            gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            if gt is None:
                print(f"Failed to read ground truth file: {gt_file}")
                skipped_count += 1
                continue

            # Check if the target label is present in the ground truth
            
            if target_label not in np.unique(gt):
                print(f"Skipping {gt_file} - target label {target_label} not found")
                skipped_count += 1
                continue

            img = cv2.imread(img_file)
            if img is None:
                print(f"Failed to read image file: {img_file}")
                skipped_count += 1
                continue

            gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

            # Only visualize the target label
            gt_rgb[gt == target_label] = color_mapping[target_label]

            mask = np.sum(gt_rgb, axis=2) > 0

            if img.shape[:2] != gt_rgb.shape[:2]:
                print(f"Mismatch in dimensions for {img_file} and {gt_file}")
                print(f"Image shape: {img.shape}, GT shape: {gt_rgb.shape}")
                skipped_count += 1
                continue

            combined_img = img.copy()
            combined_img[mask] = cv2.addWeighted(combined_img[mask], 0.4, gt_rgb[mask], 0.6, 0)

            combined_side_by_side = np.hstack((combined_img, gt_rgb))

            output_filename = os.path.join(output_dir, os.path.basename(img_file).replace('_leftImg8bit.png', f'_leftImg8bit.png'))
            cv2.imwrite(output_filename, combined_side_by_side)
            
            print(f"Processed: {img_file}")
            processed_count += 1

        except Exception as e:
            print(f"Error processing {img_file} and {gt_file}: {str(e)}")
            skipped_count += 1

    print(f"All visualized images were saved to {output_dir}")
    print(f"Processed {processed_count} images")
    print(f"Skipped {skipped_count} images")

if __name__ == '__main__':
    img_dir, gt_dir, output_dir, target_label = parse_args()
    main(img_dir, gt_dir, output_dir, target_label)

    