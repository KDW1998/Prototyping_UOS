# 균열 탐지 시스템 (Crack Detection System)

초해상화(Super Resolution)와 균열 탐지(Crack Detection)를 결합한 파이프라인입니다.

## 시스템 요구사항

- **OS**: Linux (Windows WSL2 지원)
- **GPU**: NVIDIA CUDA 지원 GPU (CUDA 11.8 이상)
- **Python**: 3.9
- **Conda**: Anaconda 또는 Miniconda

## 환경 설정

### 방법 1: Conda Environment 파일 사용 (권장)

```bash
# 1. repository 클론 또는 다운로드
cd /path/to/Prototyping_UOS

# 2. conda 환경 생성
conda env create -f environment.yml

# 3. 환경 활성화
conda activate Prototyping_UOS
```

### 방법 2: requirements.txt 사용

```bash
# 1. conda 환경 생성 (Python 3.9)
conda create -n Prototyping_UOS python=3.9

# 2. 환경 활성화
conda activate Prototyping_UOS

# 3. PyTorch 설치 (CUDA 11.8)
conda install pytorch==2.0.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 나머지 패키지 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
Prototyping_UOS/
├── 균열탐지.py                    # 메인 실행 스크립트
├── environment.yml               # Conda 환경 설정 파일
├── requirements.txt             # Python 패키지 의존성 파일
├── README.md                    # 프로젝트 설명서 (현재 파일)
├── model/                       # 모델 파일 디렉토리
│   ├── 초해상화_모델/
│   │   ├── SuperResolution_config.py
│   │   └── SuperResolution_CheckPoint.pth
│   └── 균열탐지_모델/
│       ├── CrackDetection_config.py
│       └── CrackDetection_CheckPoint.pth
├── utils/                       # 유틸리티 함수
│   ├── quantify_seg_results.py
│   └── utils.py
└── Example/                     # 예제 이미지 및 결과
    ├── 테스트이미지/
    └── 테스트이미지_탐지결과/
```

## 사용법

### 기본 사용법

```bash
# 환경 활성화
conda activate Prototyping_UOS

# 기본 경로로 실행
python 균열탐지.py
```

### 고급 사용법 (사용자 정의 경로)

```bash
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
```

### 주요 옵션

#### 입력/출력 디렉토리
- `--img-dir`: 원본 이미지가 있는 디렉토리
- `--temp-sr-dir`: 초해상화 이미지를 임시 저장할 디렉토리
- `--result-dir`: 최종 균열 탐지 결과를 저장할 디렉토리

#### 초해상화 관련
- `--sr-model-name`: 초해상화 모델 이름 (기본값: "edsr")
- `--sr-config`: 초해상화 모델 설정 파일 경로
- `--sr-checkpoint`: 초해상화 모델 체크포인트 파일 경로
- `--sr-device`: 초해상화에 사용할 디바이스 (기본값: "cuda")

#### 균열 탐지 관련
- `--crack-config`: 균열 탐지 모델 설정 파일 경로
- `--crack-checkpoint`: 균열 탐지 모델 체크포인트 파일 경로
- `--crack-device`: 균열 탐지에 사용할 디바이스 (기본값: "cuda:0")

#### 출력 설정
- `--alpha`: 균열 시각화 블렌딩 알파 값 (0.0-1.0, 기본값: 0.8)
- `--rgb-to-bgr`: RGB를 BGR로 변환 (권장)
- `--overwrite-crack-palette`: 균열 팔레트를 검은색과 빨간색으로 덮어쓰기
- `--result-suffix`: 결과 이미지 확장자 (기본값: ".JPG")
- `--mask-suffix`: 마스크 이미지 확장자 (기본값: ".png")
- `--keep-temp-sr`: 임시 초해상화 이미지 유지

## 파이프라인 설명

1. **초해상화 (Super Resolution)**
   - 입력 이미지를 고해상도로 변환
   - EDSR 모델 사용
   - 결과는 임시 디렉토리에 저장

2. **균열 탐지 (Crack Detection)**
   - 초해상화된 이미지에서 균열 탐지
   - Sliding window 방식으로 대용량 이미지 처리
   - 균열 폭과 길이 정량화

3. **결과 저장**
   - 균열이 표시된 원본 이미지
   - 균열 마스크 이미지
   - 임시 파일 정리 (옵션)

## 출력 파일

각 입력 이미지에 대해 다음 파일이 생성됩니다:

- `<이미지명>.JPG`: 균열이 시각화된 결과 이미지
- `<이미지명>.png`: 균열 마스크 이미지 (0: 배경, 1: 균열)

## 주요 의존성

- **PyTorch 2.0.1** (CUDA 11.8)
- **MMagic 1.0.1**: 초해상화 모델
- **MMSegmentation 1.0.0**: 균열 탐지 모델
- **MMCV 2.0.1**: 컴퓨터 비전 유틸리티
- **OpenCV 4.10.0**: 이미지 처리
- **NumPy 1.26.4**: 수치 연산

## 문제 해결

### CUDA Out of Memory 오류
```bash
# 배치 크기나 윈도우 크기를 줄이세요
# 또는 더 작은 이미지로 테스트하세요
```

### ImportError: No module named 'mmagic'
```bash
# 환경이 제대로 활성화되었는지 확인
conda activate Prototyping_UOS

# 패키지 재설치
pip install mmagic==1.0.1
```

### 이미지가 너무 큰 경우
```python
# 코드에 이미 적용되어 있음:
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
```

## 성능 최적화 팁

1. GPU 메모리 정리를 위해 배치 처리 후 `torch.cuda.empty_cache()` 호출 (이미 구현됨)
2. Sliding window 크기 조정 (기본값: 1024x1024)
3. Overlap ratio 조정 (기본값: 0.5)


