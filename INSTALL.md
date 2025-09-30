# 설치 가이드 (Installation Guide)

이 문서는 균열탐지 시스템을 새로운 환경에 설치하는 방법을 단계별로 설명합니다.

## 사전 요구사항

1. **Anaconda 또는 Miniconda 설치**
   - [Anaconda 다운로드](https://www.anaconda.com/download)
   - [Miniconda 다운로드](https://docs.conda.io/en/latest/miniconda.html)

2. **NVIDIA GPU 및 CUDA 드라이버**
   - NVIDIA GPU가 설치되어 있어야 합니다
   - CUDA 11.8 이상 지원 드라이버

3. **충분한 디스크 공간**
   - 최소 10GB 이상의 여유 공간 필요

## 단계별 설치

### 1단계: 프로젝트 다운로드

```bash
# Git을 사용하는 경우
git clone <repository-url>
cd Prototyping_UOS

# 또는 압축 파일을 다운로드한 경우
unzip Prototyping_UOS.zip
cd Prototyping_UOS
```

### 2단계: Conda 환경 생성

#### 옵션 A: environment.yml 사용 (권장)

```bash
# 환경 생성 (모든 패키지가 자동으로 설치됩니다)
conda env create -f environment.yml

# 환경 활성화
conda activate Prototyping_UOS
```

#### 옵션 B: 수동 설치

```bash
# 1. Python 3.9 환경 생성
conda create -n Prototyping_UOS python=3.9 -y

# 2. 환경 활성화
conda activate Prototyping_UOS

# 3. PyTorch 및 관련 패키지 설치
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 기타 conda 패키지 설치
conda install numpy=1.26.4 pillow ffmpeg -c conda-forge -y

# 5. pip 패키지 설치
pip install -r requirements.txt
```

### 3단계: 환경 확인

```bash
# 환경이 제대로 설치되었는지 확인
python check_environment.py
```

정상적으로 설치되었다면 다음과 같은 메시지가 표시됩니다:
```
✓ 모든 확인 항목이 통과되었습니다!
균열탐지 시스템을 사용할 준비가 완료되었습니다.
```

### 4단계: 모델 파일 확인

모델 파일이 다음 위치에 있는지 확인하세요:

```
model/
├── 초해상화_모델/
│   ├── SuperResolution_config.py
│   └── SuperResolution_CheckPoint.pth
└── 균열탐지_모델/
    ├── CrackDetection_config.py
    └── CrackDetection_CheckPoint.pth
```

모델 파일이 없다면 담당자에게 문의하여 받으세요.

### 5단계: 테스트 실행

```bash
# 기본 설정으로 테스트 실행
python 균열탐지.py --rgb-to-bgr

# 또는 예제 이미지가 있다면
python 균열탐지.py \
    --img-dir "Example/테스트이미지" \
    --result-dir "Example/테스트이미지_탐지결과" \
    --rgb-to-bgr
```

## 문제 해결

### ImportError: No module named 'xxx'

특정 패키지가 설치되지 않은 경우:

```bash
# 개별 패키지 설치
pip install <package-name>

# 또는 requirements.txt 재설치
pip install -r requirements.txt
```

### CUDA 관련 오류

```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 버전 확인
nvidia-smi

# PyTorch 재설치 (CUDA 11.8)
pip uninstall torch torchvision torchaudio
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Conda 환경 생성 실패

```bash
# 기존 환경 삭제 후 재생성
conda env remove -n Prototyping_UOS
conda env create -f environment.yml
```

### MMagic, MMSeg 설치 오류

```bash
# MMCV 먼저 설치
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 그 다음 MMagic, MMSeg 설치
pip install mmagic==1.0.1
pip install mmsegmentation==1.0.0
```

## 환경 업데이트

새로운 패키지가 추가된 경우:

```bash
# 환경 활성화
conda activate Prototyping_UOS

# environment.yml 업데이트
conda env update -f environment.yml --prune

# 또는 requirements.txt 업데이트
pip install -r requirements.txt --upgrade
```

## 환경 백업 및 공유

현재 환경을 다른 사람과 공유하거나 백업하려면:

```bash
# Conda 환경 export
conda env export > environment.yml

# Pip 패키지 목록 export
pip list --format=freeze > requirements.txt
```

## 환경 삭제

더 이상 사용하지 않는 경우:

```bash
# 환경 비활성화
conda deactivate

# 환경 삭제
conda env remove -n Prototyping_UOS
```

## 지원

설치 중 문제가 발생하면:

1. `check_environment.py`를 실행하여 어떤 부분이 문제인지 확인
2. 오류 메시지를 복사하여 담당자에게 문의
3. 시스템 정보 제공:
   ```bash
   # OS 정보
   uname -a
   
   # CUDA 정보
   nvidia-smi
   
   # Python 버전
   python --version
   
   # Conda 정보
   conda info
   ```

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [MMagic 문서](https://mmagic.readthedocs.io/)
- [MMSegmentation 문서](https://mmsegmentation.readthedocs.io/)
- [Conda 사용자 가이드](https://docs.conda.io/projects/conda/en/latest/user-guide/)
