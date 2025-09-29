'''
python 균열탐지.py --srx_dir 'Example/테스트이미지' --rst_dir 'Example/테스트이미지_탐지결과' --sr_model_name 'edsr' --crack_config 'model/균열탐지_모델/CrackDetection_config.py' --crack_checkpoint 'model/균열탐지_모델/CrackDetection_CheckPoint.pth' --alpha 0.6 --crack_label 1 --device 'cuda:0'
'''
import argparse
import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path

# OpenMMLab 라이브러리
from mmagic.apis import MMagicInferencer
from mmseg.apis import init_model
import mmcv

# 로컬 유틸리티 (필요 시, 이전 코드의 utils.py에서 가져와야 함)
# 이 파일이 없으면 이 라인을 주석 처리하고, sliding window 추론 부분을 일반 추론으로 변경해야 합니다.
from utils.utils import inference_segmentor_sliding_window 

# ==============================================================================
# 상수 정의 (Define Constants)
# ==============================================================================

# 시각화를 위한 색상 매핑 (BGR 순서)
# 균열(레이블 1)은 빨간색으로 표시
color_mapping = {
    0: [0, 0, 0],       # Class 0: 배경 (검은색)
    1: [0, 0, 255],     # Class 1: 균열 (빨간색)
}

# ==============================================================================
# 인자 파싱 함수 (Argument Parsing Function)
# ==============================================================================
def parse_args():
    """커맨드 라인 인자를 파싱하여 반환합니다."""
    parser = argparse.ArgumentParser(description='End-to-End Crack Detection Pipeline: Super-Resolution -> Detection -> Visualization')

    # --- 입력 및 출력 경로 인자 ---
    parser.add_argument('--srx_dir', required=True, help='처리할 원본 이미지가 있는 디렉토리')
    parser.add_argument('--rst_dir', required=True, help='최종 시각화 결과물을 저장할 디렉토리')
    parser.add_argument('--srx_suffix', default='.png', help='처리할 원본 이미지의 확장자')

    # --- 1. 초해상화(Super-Resolution) 모델 인자 ---
    parser.add_argument('--sr_model_name', type=str, default='edsr', help='초해상화에 사용할 MMagic 모델 이름 (예: esrgan, rdn)')

    # --- 2. 균열 탐지(Crack Detection) 모델 인자 ---
    parser.add_argument('--crack_config', required=True, help='균열 탐지 모델의 설정(config) 파일')
    parser.add_argument('--crack_checkpoint', required=True, help='균열 탐지 모델의 체크포인트(checkpoint) 파일')

    # --- 3. 시각화(Visualization) 인자 ---
    parser.add_argument('--alpha', type=float, default=0.6, help='마스크를 이미지에 오버레이할 때의 투명도 값')
    parser.add_argument('--crack_label', type=int, default=1, help='마스크에서 균열을 나타내는 레이블 인덱스')
    
    # --- 공통 인자 ---
    parser.add_argument('--device', type=str, default='cuda:0', help='추론에 사용할 장치 (예: "cuda:0" 또는 "cpu")')

    args = parser.parse_args()
    return args

# ==============================================================================
# 메인 실행 함수 (Main Function)
# ==============================================================================
def main():
    """메인 파이프라인을 실행합니다."""
    args = parse_args()

    # --- 디렉토리 설정 ---
    # 최종 결과물 외에, 중간 과정(초해상도, 마스크)을 저장할 임시 디렉토리 생성
    sr_output_dir = os.path.join(args.rst_dir, 'temp_super_resolution')
    Path(args.rst_dir).mkdir(parents=True, exist_ok=True)
    Path(sr_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"1. 모델을 초기화합니다...")
    # 1. 초해상화 모델 초기화
    sr_inferencer = MMagicInferencer(model_name=args.sr_model_name, device=args.device)
    
    # 2. 균열 탐지 모델 초기화
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device=args.device)
    print("모델 초기화 완료.")

    # --- 입력 이미지 목록 가져오기 ---
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))
    if not img_list:
        print(f"오류: '{args.srx_dir}' 디렉토리에서 '{args.srx_suffix}' 확장자를 가진 이미지를 찾을 수 없습니다.")
        return
        
    print(f"총 {len(img_list)}개의 이미지를 처리합니다.")

    # --- 각 이미지에 대해 파이프라인 실행 ---
    for img_path in img_list:
        base_name = os.path.basename(img_path)
        print(f"\n--- [{base_name}] 처리 시작 ---")

        # ==========================
        # 단계 1: 초해상화 (Super-Resolution)
        # ==========================
        print(f"  [1/3] 초해상화 진행 중...")
        sr_img_path = os.path.join(sr_output_dir, base_name)
        sr_inferencer.infer(img=img_path, result_out_dir=sr_output_dir)
        # MMagicInferencer는 원본 파일명으로 저장하므로, 저장된 경로를 다시 확인합니다.
        
        # 초해상화된 이미지 로드
        sr_image = cv2.imread(sr_img_path)
        if sr_image is None:
            print(f"    경고: 초해상화된 이미지 파일을 읽는 데 실패했습니다: {sr_img_path}")
            continue

        # ==========================
        # 단계 2: 균열 탐지 (Crack Detection)
        # ==========================
        print(f"  [2/3] 균열 탐지 진행 중...")
        # Sliding window 방식으로 추론하여 메모리 부족 문제 방지
        _, crack_mask = inference_segmentor_sliding_window(
            crack_model, 
            sr_img_path, 
            color_mask=None, 
            score_thr=0.5,
            window_size=1024, 
            overlap_ratio=1
        )

        # ==========================
        # 단계 3: 시각화 및 저장 (Visualization & Save)
        # ==========================
        print(f"  [3/3] 결과 시각화 및 저장 중...")
        
        # 3-1. 균열 마스크를 RGB 색상 마스크로 변환
        gt_rgb = np.zeros((crack_mask.shape[0], crack_mask.shape[1], 3), dtype=np.uint8)
        gt_rgb[crack_mask == args.crack_label] = color_mapping[args.crack_label]
        
        # 3-2. 원본(초해상화된) 이미지에 색상 마스크 오버레이
        mask_bool = np.sum(gt_rgb, axis=2) > 0
        combined_img = sr_image.copy()
        combined_img[mask_bool] = cv2.addWeighted(
            combined_img[mask_bool], 1 - args.alpha, 
            gt_rgb[mask_bool], args.alpha, 
            0
        )
        
        # 3-3. [오버레이 이미지]와 [색상 마스크]를 나란히 붙여 최종 결과물 생성
        combined_side_by_side = np.hstack((combined_img, gt_rgb))

        # 3-4. 최종 결과물 저장
        output_filename = os.path.join(args.rst_dir, base_name)
        cv2.imwrite(output_filename, combined_side_by_side)
        print(f"    -> 최종 결과 저장 완료: {output_filename}")

    print("\n모든 작업이 완료되었습니다.")
    print(f"최종 결과는 '{args.rst_dir}' 디렉토리에 저장되었습니다.")


# ==============================================================================
# 스크립트 실행 지점 (Script Entry Point)
# ==============================================================================
if __name__ == '__main__':
    main()