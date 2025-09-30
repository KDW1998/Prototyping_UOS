#!/usr/bin/env python
"""
환경 검증 스크립트
이 스크립트는 균열탐지 시스템에 필요한 모든 패키지가 올바르게 설치되었는지 확인합니다.
"""

import sys

def check_python_version():
    """Python 버전 확인"""
    print("=" * 70)
    print("Python 버전 확인")
    print("=" * 70)
    version = sys.version_info
    print(f"현재 Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 9:
        print("✓ Python 버전이 올바릅니다.")
        return True
    else:
        print("✗ Python 3.9가 필요합니다.")
        return False

def check_package(package_name, import_name=None, version_attr='__version__'):
    """패키지 설치 확인"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, "버전 정보 없음")
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: 설치되지 않음")
        return False
    except Exception as e:
        print(f"✗ {package_name}: 오류 발생 - {str(e)}")
        return False

def check_cuda():
    """CUDA 사용 가능 여부 확인"""
    print("\n" + "=" * 70)
    print("CUDA 및 GPU 확인")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("✓ CUDA가 올바르게 설정되었습니다.")
            return True
        else:
            print("✗ CUDA를 사용할 수 없습니다. GPU가 없거나 CUDA가 올바르게 설치되지 않았습니다.")
            return False
    except ImportError:
        print("✗ PyTorch가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print(f"✗ CUDA 확인 중 오류 발생: {str(e)}")
        return False

def check_main_packages():
    """주요 패키지 확인"""
    print("\n" + "=" * 70)
    print("주요 패키지 확인")
    print("=" * 70)
    
    packages = [
        ('torch', 'torch', '__version__'),
        ('torchvision', 'torchvision', '__version__'),
        ('numpy', 'numpy', '__version__'),
        ('opencv-python', 'cv2', '__version__'),
        ('mmcv', 'mmcv', '__version__'),
        ('mmseg', 'mmseg', '__version__'),
        ('mmagic', 'mmagic', '__version__'),
        ('mmdet', 'mmdet', '__version__'),
        ('mmengine', 'mmengine', '__version__'),
        ('pillow', 'PIL', '__version__'),
    ]
    
    results = []
    for package_name, import_name, version_attr in packages:
        results.append(check_package(package_name, import_name, version_attr))
    
    return all(results)

def check_model_files():
    """모델 파일 존재 여부 확인"""
    print("\n" + "=" * 70)
    print("모델 파일 확인")
    print("=" * 70)
    
    import os
    
    model_files = [
        'model/초해상화_모델/SuperResolution_config.py',
        'model/초해상화_모델/SuperResolution_CheckPoint.pth',
        'model/균열탐지_모델/CrackDetection_config.py',
        'model/균열탐지_모델/CrackDetection_CheckPoint.pth',
    ]
    
    results = []
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file_path} ({size:.2f} MB)")
            results.append(True)
        else:
            print(f"✗ {file_path} - 파일을 찾을 수 없습니다")
            results.append(False)
    
    return all(results)

def check_utils():
    """유틸리티 모듈 확인"""
    print("\n" + "=" * 70)
    print("유틸리티 모듈 확인")
    print("=" * 70)
    
    try:
        from utils.quantify_seg_results import quantify_crack_width_length
        print("✓ utils.quantify_seg_results 모듈을 불러왔습니다.")
        result1 = True
    except ImportError as e:
        print(f"✗ utils.quantify_seg_results 모듈을 불러올 수 없습니다: {str(e)}")
        result1 = False
    
    try:
        from utils.utils import inference_segmentor_sliding_window
        print("✓ utils.utils 모듈을 불러왔습니다.")
        result2 = True
    except ImportError as e:
        print(f"✗ utils.utils 모듈을 불러올 수 없습니다: {str(e)}")
        result2 = False
    
    return result1 and result2

def main():
    """메인 함수"""
    print("\n")
    print("*" * 70)
    print("균열탐지 시스템 환경 검증")
    print("*" * 70)
    print("\n")
    
    results = []
    
    # Python 버전 확인
    results.append(check_python_version())
    
    # 주요 패키지 확인
    results.append(check_main_packages())
    
    # CUDA 확인
    results.append(check_cuda())
    
    # 모델 파일 확인
    results.append(check_model_files())
    
    # 유틸리티 모듈 확인
    results.append(check_utils())
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)
    
    if all(results):
        print("✓ 모든 확인 항목이 통과되었습니다!")
        print("균열탐지 시스템을 사용할 준비가 완료되었습니다.")
        return 0
    else:
        print("✗ 일부 확인 항목이 실패했습니다.")
        print("위의 오류 메시지를 확인하고 필요한 패키지를 설치하세요.")
        print("\n환경 설정 방법:")
        print("  conda env create -f environment.yml")
        print("  conda activate Prototyping_UOS")
        return 1

if __name__ == '__main__':
    sys.exit(main())
