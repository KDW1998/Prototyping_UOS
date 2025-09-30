'''
Combined Script: Super Resolution + Crack Detection

Usage (with default paths):
    python 균열탐지.py --rgb-to-bgr

Usage (with custom paths):
    python 균열탐지.py \
        --img-dir "Example/테스트이미지" \
        --temp-sr-dir "Example/테스트이미지_탐지결과/temp_super_resolution" \
        --result-dir "Example/테스트이미지_탐지결과" \
        --sr-model-name "edsr" \
        --sr-config "model/초해상화_모델/SuperResolution_config.py" \
        --sr-checkpoint "model/초해상화_모델/SuperResolution_CheckPoint.pth" \
        --crack-config "model/균열탐지_모델/CrackDetection_config.py" \
        --crack-checkpoint "model/균열탐지_모델/CrackDetection_CheckPoint.pth" \
        --rgb-to-bgr
'''

import os
from pickle import TRUE
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import argparse
from glob import glob
import shutil

# Super Resolution imports
from mmagic.apis import MMagicInferencer

# Crack Detection imports
from mmseg.apis import init_model
import mmcv
import numpy as np
from utils.quantify_seg_results import quantify_crack_width_length
from torch.cuda import empty_cache
from utils.utils import inference_segmentor_sliding_window


def parse_args():
    parser = argparse.ArgumentParser(description='Combined Super Resolution and Crack Detection')
    
    # Input/Output directories
    parser.add_argument('--img-dir', type=str, default='Example/테스트이미지',
                        help='Input directory containing original images')
    parser.add_argument('--temp-sr-dir', type=str, default='Example/테스트이미지_탐지결과/temp_super_resolution',
                        help='Temporary directory for super-resolved images')
    parser.add_argument('--result-dir', type=str, default='Example/테스트이미지_탐지결과',
                        help='Final output directory for crack detection results')
    
    # Super Resolution parameters
    parser.add_argument('--sr-model-name', type=str, default='edsr',
                        help='Super resolution model name')
    parser.add_argument('--sr-config', type=str, default='model/초해상화_모델/SuperResolution_config.py',
                        help='Super resolution model config file')
    parser.add_argument('--sr-checkpoint', type=str, default='model/초해상화_모델/SuperResolution_CheckPoint.pth',
                        help='Super resolution model checkpoint file')
    parser.add_argument('--sr-device', type=str, default='cuda',
                        help='Device for super resolution')
    
    # Crack Detection parameters
    parser.add_argument('--crack-config', type=str, default='model/균열탐지_모델/CrackDetection_config.py',
                        help='Crack detection model config file')
    parser.add_argument('--crack-checkpoint', type=str, default='model/균열탐지_모델/CrackDetection_CheckPoint.pth',
                        help='Crack detection model checkpoint file')
    parser.add_argument('--crack-device', type=str, default='cuda:0',
                        help='Device for crack detection')
    
    # Output parameters
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Alpha value for blending crack visualization')
    parser.add_argument('--rgb-to-bgr', action='store_true',
                        help='Convert RGB to BGR for crack palette')
    parser.add_argument('--overwrite-crack-palette', action='store_true',
                        help='Overwrite crack palette with black and red')
    parser.add_argument('--result-suffix', type=str, default='.JPG',
                        help='Suffix for result images')
    parser.add_argument('--mask-suffix', type=str, default='.png',
                        help='Suffix for mask output')
    parser.add_argument('--keep-temp-sr', action='store_true',
                        help='Keep temporary super-resolved images after processing')
    
    parser.set_defaults(rgb_to_bgr=False, overwrite_crack_palette=False, keep_temp_sr=False)
    
    args = parser.parse_args()
    return args


def super_resolution_step(args):
    """
    Step 1: Apply super resolution to all images in the input directory
    """
    print("\n" + "="*80)
    print("STEP 1: Super Resolution Processing")
    print("="*80)
    
    # Create temporary directory for super-resolved images
    os.makedirs(args.temp_sr_dir, exist_ok=True)
    
    # Initialize super resolution inferencer
    print(f"Initializing super resolution model: {args.sr_model_name}")
    sr_inferencer = MMagicInferencer(
        model_name=args.sr_model_name,
        model_config=args.sr_config,
        model_ckpt=args.sr_checkpoint,
        device=args.sr_device
    )
    
    # Get list of images to process
    img_list = []
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        img_list.extend(glob(os.path.join(args.img_dir, f'*{ext}')))
    
    print(f"Found {len(img_list)} images to process")
    
    # Process each image
    for idx, img_path in enumerate(img_list, 1):
        img_name = os.path.basename(img_path)
        # Change extension to .png for super-resolved images
        sr_img_name = os.path.splitext(img_name)[0] + '.png'
        output_path = os.path.join(args.temp_sr_dir, sr_img_name)
        
        print(f"[{idx}/{len(img_list)}] Processing: {img_name}")
        
        # Run super resolution
        sr_inferencer.infer(
            img=img_path,
            result_out_dir=output_path
        )
        
        # Clear GPU cache
        empty_cache()
    
    print(f"Super resolution complete. Results saved to: {args.temp_sr_dir}\n")
    return len(img_list)


def crack_detection_step(args):
    """
    Step 2: Apply crack detection to super-resolved images
    """
    print("\n" + "="*80)
    print("STEP 2: Crack Detection Processing")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Initialize crack detection model
    print("Initializing crack detection model...")
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device=args.crack_device)
    
    # Get list of super-resolved images
    img_list = glob(os.path.join(args.temp_sr_dir, '*.png'))
    print(f"Found {len(img_list)} super-resolved images to analyze")
    
    # Setup crack palette
    crack_palette = crack_model.dataset_meta['palette'][:2]
    if args.rgb_to_bgr:
        crack_palette = [p[::-1] for p in crack_palette]
    if args.overwrite_crack_palette:
        crack_palette[1] = [0, 0, 255]
    
    # Process each super-resolved image
    for idx, img_path in enumerate(img_list, 1):
        img_name = os.path.basename(img_path)
        print(f"[{idx}/{len(img_list)}] Detecting cracks: {img_name}")
        
        # Run crack detection
        _, crack_mask = inference_segmentor_sliding_window(
            crack_model, 
            img_path, 
            color_mask=None, 
            score_thr=0.5, 
            window_size=1024, 
            overlap_ratio=0.5
        )
        
        # Load original image
        seg_result = mmcv.imread(img_path)
        
        # Visualize crack mask
        color = np.array(crack_palette[1], dtype=np.uint8)
        mask_bool = crack_mask == 1
        seg_result[mask_bool, :] = seg_result[mask_bool, :] * (1 - args.alpha) + color * args.alpha
        
        # Prepare output names
        base_name = os.path.splitext(img_name)[0]
        rst_name = base_name + args.result_suffix
        mask_name = base_name + args.mask_suffix
        
        # Quantify crack width and length
        seg_result = quantify_crack_width_length(seg_result, crack_mask, crack_palette[1])
        
        # Save results
        rst_path = os.path.join(args.result_dir, rst_name)
        mask_path = os.path.join(args.result_dir, mask_name)
        
        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(crack_mask.astype(np.uint8), mask_path)
        
        # Clear GPU cache
        empty_cache()
    
    print(f"Crack detection complete. Results saved to: {args.result_dir}\n")
    return len(img_list)


def cleanup_temp_files(args):
    """
    Step 3: Clean up temporary super-resolved images if requested
    """
    if not args.keep_temp_sr and os.path.exists(args.temp_sr_dir):
        print("\n" + "="*80)
        print("STEP 3: Cleaning up temporary files")
        print("="*80)
        print(f"Removing temporary directory: {args.temp_sr_dir}")
        shutil.rmtree(args.temp_sr_dir)
        print("Cleanup complete.\n")
    elif args.keep_temp_sr:
        print(f"\nTemporary super-resolved images kept at: {args.temp_sr_dir}\n")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("COMBINED CRACK DETECTION PIPELINE")
    print("="*80)
    print(f"Input directory:  {args.img_dir}")
    print(f"Temp SR directory: {args.temp_sr_dir}")
    print(f"Output directory: {args.result_dir}")
    print("="*80 + "\n")
    
    try:
        # Step 1: Super Resolution
        num_images = super_resolution_step(args)
        
        # Step 2: Crack Detection
        num_processed = crack_detection_step(args)
        
        # Step 3: Cleanup (if requested)
        cleanup_temp_files(args)
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total images processed: {num_images}")
        print(f"Results available at: {args.result_dir}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
