# KolektorSDD 표면 결함 분류 (Surface Defect Classification)

> KolektorSDD 산업용 표면 결함 데이터셋을 활용한 머신비전 이진 분류 프로젝트
> 데이터 파이프라인 개선과 전이학습 적용을 통한 점진적 성능 향상 과정을 기록했습니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---


## 목차 (Table of Contents)

1. [프로젝트 개요](#프로젝트-개요)
2. [데이터셋](#데이터셋)
3. [프로젝트 구조](#프로젝트-구조)
4. [설치 및 실행](#설치-및-실행)
5. [결과](#결과)
   - [탐색적 데이터 분석](#탐색적-데이터-분석)
   - [베이스라인 모델](#베이스라인-모델)
   - [데이터 중심 개선](#데이터-중심-개선)
   - [모델 중심 개선](#모델-중심-개선)
6. [개발 로드맵](#개발-로드맵)
7. [라이센스](#라이센스)

---


## 프로젝트 개요

KolektorSDD 산업용 표면 결함 데이터셋을 활용한 이진 분류 머신비전 프로젝트입니다. 총 399장의 이미지 중 52장(결함)과 347장(정상)으로 구성된 클래스 불균형 환경에서, 데이터 중심 개선 과정과 모델 중심 개선 과정을 통해 결함 탐지 성능을 높이는 것을 목표로 합니다.

본 프로젝트에서는 분류 성능 지표(F1, AUC)와 더불어, 미탐(FN)이 오탐(FP)보다 더 큰 비용을 유발할 수 있다는 가정 하에 비용 시뮬레이션을 함께 고려했습니다.

비용 수치는 실제 산업 데이터가 아닌 품질 리스크를 단순화하여 설정한 가정값이며, 프로젝트 내 의사결정 기준 정의를 위한 목적으로 사용되었습니다.

Total Cost는 아래와 같이 정의했습니다.

**Total Cost = FN x C_FN + FP x C_FP**

모델 성능 개선은 F1, AUC 지표를 기준으로 평가를 진행하며, FN/FP 비용 비대칭성을 반영한 threshold 최적화를 통해 Total Cost 최소화를 함께 고려하는 방향으로 접근하고자 합니다.

---


## 데이터셋

### KolektorSDD

| 항목 | 내용 |
|---|---|
| 전체 이미지 | 399장 |
| 결함 | 52장 |
| 정상 | 347장 |
| 불균형 비율 | ~1:6.7 (결함:정상) |
| 파트 수 | 50개 |
| 이미지 크기 범위 | 500x1235 ~ 500x1285 픽셀 |


> **출처 및 저작권:**
- 논문: "Segmentation-based deep-learning approach for surface-defect detection"
- 데이터셋 공식 페이지: https://www.vicos.si/resources/kolektorsdd/
- 라이센스: **CC BY-NC-SA 4.0**


> **사용 안내:**
- 본 프로젝트는 **개인 포트폴리오 목적**으로 데이터셋을 사용합니다.
- 원본 데이터 파일은 저장소에 포함되지 않습니다 (`.gitignore` 처리).
- 데이터셋은 위 공식 출처에서 직접 다운로드하시기 바랍니다.

---


## 프로젝트 구조 (Structure)

```
kolektorsdd-classification/
├── configs/
│   ├── base.yaml                        # 공통 설정
│   ├── experiment_baseline.yaml         # 베이스라인 실험 설정
│   ├── experiment_data_centric.yaml     # 데이터 중심 개선 실험 설정
│   └── experiment_model_centric.yaml    # 모델 중심 개선 실험 설정
├── data/
│   ├── raw/                             # 원본 데이터 (gitignore)
│   └── processed/
│       ├── train.csv                    # 학습 분할 (Split)
│       ├── val.csv                      # 검증 분할
│       └── test.csv                     # 테스트 분할
├── experiments/
│   ├── baseline/
│   │   ├── results.json                 # 베이스라인 실험 결과
│   │   └── figures/                     # 학습 곡선 및 평가 시각화
│   ├── data_centric/
│   │   ├── results.json                 # 데이터 중심 개선 실험 결과
│   │   └── figures/                     # 학습 곡선 및 평가 시각화
│   └── model_centric/
│       ├── results.json                 # 모델 중심 개선 실험 결과
│       └── figures/                     # 학습 곡선, 평가 곡선, Grad-CAM 시각화
├── notebooks/
│   ├── 01_eda.ipynb                     # 탐색적 데이터 분석 노트북
│   ├── 02_baseline.ipynb                # 베이스라인 모델 노트북
│   ├── 03_data_centric.ipynb            # 데이터 중심 개선 노트북
│   └── 04_model_centric.ipynb           # 모델 중심 개선 노트북
├── reports/
│   └── figures/                         # EDA 시각화 이미지
├── src/
│   ├── data/                            # 데이터셋, 전처리, 증강, 샘플러
│   ├── models/                          # 모델 정의 (CNN, EfficientNet-B0 전이학습)
│   ├── training/                        # 학습 루프, 손실 함수
│   ├── evaluation/                      # 평가 지표
│   └── utils/                           # 유틸리티
├── requirements.txt
└── pyproject.toml
```

---


## 설치 및 실행

### 1. 환경 설정

```bash
git clone https://github.com/shinnhyuk/kolektorsdd-classification.git
cd kolektorsdd-classification
pip install -r requirements.txt
```

### 2. 데이터셋 준비

KolektorSDD 데이터셋을 [공식 페이지](https://www.vicos.si/resources/kolektorsdd/)에서 다운로드한 후 아래 경로에 배치합니다.

```
data/raw/KolektorSDD/
├── kos01/
│   ├── Part0.jpg
│   ├── Part0_label.bmp
│   └── ...
├── kos02/
└── ...
```

---


## 결과

### 탐색적 데이터 분석

데이터셋 분석 후, raw data의 폴더 파트 단위로 stratified split을 수행했습니다. 동일 파트의 이미지가 train/val/test에 중복 포함되지 않도록 하여 데이터 누수를 방지했습니다.

> **픽셀 통계 (train 기준):**

| 항목 | 값 |
|---|---|
| 평균 (Mean) | 0.7317 |
| 표준편차 (Std) | 0.0915 |
| 이미지 크기 범위 | 500x1235 ~ 500x1285 |


> **데이터 분할 결과:**

| 분할 | 전체 | 결함 | 정상 |
|---|---|---|---|
| train | 280장 | 36장 | 244장 |
| val | 56장 | 7장 | 49장 |
| test | 63장 | 9장 | 54장 |


> **EDA 시각화:**

클래스 분포:

![클래스 분포](reports/figures/class_distribution.png)

정상 샘플:

![정상 샘플](reports/figures/normal_samples.png)

결함 샘플:

![결함 샘플](reports/figures/defect_samples.png)

픽셀 강도 분포:

![픽셀 히스토그램](reports/figures/pixel_histogram.png)

---


### 베이스라인 모델

최소한의 증강 (수평 플립, 리사이즈, 정규화)만 적용한 Simple Custom CNN으로 초기 성능 베이스라인을 수립했습니다. 클래스 불균형 대응을 위해 BCEWithLogitsLoss에 pos_weight를 자동 계산하여 적용했습니다.


> **모델 구성:**

| 항목 | 값 |
|---|---|
| 아키텍처 | Simple Custom CNN (3-block) |
| 입력 크기 | 256x256 |
| 손실 함수 | BCEWithLogitsLoss (pos_weight 자동계산) |
| 옵티마이저 | Adam (lr=0.001) |
| 스케줄러 | CosineAnnealing |
| 에포크 | 50 (Early Stopping patience=10) |


> **평가 결과 (threshold=0.5):**

| 분할 | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| val | 0.2727 | 0.5569 | 0.1622 | 0.8571 |
| test | 0.2632 | 0.5535 | 0.1724 | 0.5556 |

> 높은 재현율과 낮은 정밀도가 공존하는 전형적인 불균형 초기 결과로 나타남. 이후 데이터 파이프라인 및 모델 개선을 통해 F1과 AUC-ROC를 함께 끌어올리는 것이 목표.


> **실패 케이스 분석 (test set 기준):**

| 구분 | 건수 | 추정 비용 (가정치) |
|---|---|---|
| FN | 4건 | 50,000,000원/건 → 200,000,000원 |
| FP | 24건 | 500,000원/건 → 12,000,000원 |
| **합계** | | **212,000,000원** |

> FN 비용(50,000,000원/건)은 결함 1건이 품질 검사 통과로 출하될 경우 발생할 수 있는 리콜·보상·라인 중단 등 하류 리스크를 금전적으로 환산한 잠재 비용을 보수적으로 추정한 값. 본 프로젝트에서는 모델 비교 및 threshold 최적화를 위해 FN/FP 비용을 건별 비용으로 단순화하여 Total Cost를 계산함.

- **FN 주원인:** 원본 이미지 세로 방향 압축으로 인한 결함 소실 / 학습 결함 데이터 부족(36장) / threshold=0.5 부적절
- **FP 주원인:** 표면 노이즈·조명 불균일 오인 / 결함 인접 정상 이미지 혼동
- **개선 방향:** Augmentation 강화 + 클래스 불균형 보정 강화 → 전이학습 → threshold 최적화


> **학습 곡선 및 평가 시각화:**

학습 이력:

![학습 이력](experiments/baseline/figures/training_history.png)

평가 곡선:

![평가 곡선](experiments/baseline/figures/evaluation_curves.png)

샘플 이미지:

![샘플 이미지](experiments/baseline/figures/sample_images.png)

---


### 데이터 중심 개선

베이스라인과 동일한 CNN 아키텍처를 유지하면서, 데이터 파이프라인 관점에서의 개선을 집중 적용했습니다. 금속 표면 특성에 특화된 증강 기법과 클래스 불균형 보정을 강화하여 AUC-ROC 및 AP 지표를 중심으로 개선을 달성했습니다.


> **적용 기법:**

| 기법 | 설정 | 목적 |
|---|---|---|
| CLAHE | clip_limit=2.0, p=0.5 | 금속 표면 국소 대비 강화 |
| RandomBrightnessContrast | p=0.4 | 조명 변화 대응 |
| HorizontalFlip | p=0.5 | 좌우 대칭 불변성 확보 |
| BCEWithLogitsLoss + pos_weight | pos_weight≈6.78 | 클래스 불균형 보정 |


> **검토 후 미적용 기법:**

| 기법 | 제외 사유 |
|---|---|
| 레터박스 리사이즈 | 256×256 입력 중 실제 이미지 영역이 ~40%에 불과해 베이스라인 CNN이 패딩 영역을 배경으로 학습하지 못하고 오경보 증가. 전이학습 모델 적용 시 재검토 예정. |
| WeightedRandomSampler | pos_weight와 동시 적용 시 이중 불균형 보정으로 FP 폭증 |
| ElasticTransform | 얇은 선형 결함 패턴을 왜곡하며, 훈련 데이터 36장 규모에서는 정규화 효과보다 결함 특징 소실이 더 큼 |
| FocalLoss | alpha 설정 방향이 데이터셋 불균형 구조와 맞지 않아 불안정 학습 발생. BCEWithLogitsLoss + pos_weight가 더 안정적 |


> **평가 결과 (threshold=0.5):**

| 분할 | F1 | AUC-ROC | AP | Precision | Recall |
|---|---|---|---|---|---|
| val | 0.2609 | 0.5539 | 0.1538 | 0.1538 | 0.8571 |
| test | 0.2642 | 0.6008 | 0.3668 | 0.1591 | 0.7778 |


> **베이스라인 대비 변화 (test set):**

| 지표 | 베이스라인 | 데이터 중심 개선 | 변화 |
|---|---|---|---|
| F1 | 0.2632 | 0.2642 | +0.0010 |
| AUC-ROC | 0.5535 | 0.6008 | **+0.0473** |
| AP | 0.2720 | 0.3668 | **+0.0948** |
| Recall | 0.5556 | 0.7778 | **+0.2222** |
| FN 건수 | 4건 | 2건 | **-2건** |
| FP 건수 | 24건 | 37건 | +13건 |

> threshold=0.5 기준 F1은 거의 변화가 없으나, AUC-ROC(+0.047)와 AP(+0.095)가 소폭 향상되었고 재현율(Recall)이 0.556에서 0.778로 높아지면서 FN이 4건에서 2건으로 감소했습니다. 단, FP가 24건에서 37건으로 증가하여 F1 개선으로 이어지지 않았습니다. 이후 threshold 최적화를 통해 FN/FP 균형을 조정할 수 있습니다.

> **학습 곡선 및 평가 시각화:**

학습 이력:

![학습 이력](experiments/data_centric/figures/training_history.png)

평가 곡선:

![평가 곡선](experiments/data_centric/figures/evaluation_curves.png)

샘플 이미지:

![샘플 이미지](experiments/data_centric/figures/sample_images.png)

---


### 모델 중심 개선

데이터 파이프라인(증강, pos_weight)을 그대로 유지하면서, 아키텍처를 Custom CNN에서 EfficientNet-B0 (ImageNet 사전학습) 전이학습 모델로 교체했습니다. 2단계 파인튜닝 (Frozen Epochs 5 → 전체 파인튜닝) 전략을 적용하여 특징 추출 능력을 활용했습니다.


> **모델 구성:**

| 항목 | 값 |
|---|---|
| 아키텍처 | EfficientNet-B0 (ImageNet 사전학습) |
| 입력 크기 | 256x256 |
| 파인튜닝 전략 | 2단계 (헤드 학습 5 에포크 → 전체 파인튜닝) |
| 정규화 | ImageNet 통계 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| 손실 함수 | BCEWithLogitsLoss (pos_weight 자동계산) |
| 옵티마이저 | Adam |
| 스케줄러 | CosineAnnealing |


> **평가 결과 (threshold=0.5):**

| 분할 | F1 | AUC-ROC | AP | Precision | Recall |
|---|---|---|---|---|---|
| val | 0.8571 | 0.9942 | 0.9683 | 0.8571 | 0.8571 |
| test | 0.7000 | 0.9691 | 0.8959 | 0.6364 | 0.7778 |

> **데이터 중심 개선 대비 변화 (test set):**

| 지표 | 데이터 중심 개선 | 모델 중심 개선 | 변화 |
|---|---|---|---|
| F1 | 0.2642 | 0.7000 | **+0.4358** |
| AUC-ROC | 0.6008 | 0.9691 | **+0.3683** |
| AP | 0.3668 | 0.8959 | **+0.5291** |
| FN 건수 | 2건 | 2건 | ±0건 |
| FP 건수 | 37건 | 4건 | **-33건** |

> 전이학습 도입으로 F1이 0.264에서 0.700으로, AUC-ROC가 0.601에서 0.969로 대폭 향상되었습니다. FP가 37건에서 4건으로 크게 감소하면서 정밀도가 개선된 것이 핵심입니다. FN은 동일하게 2건으로 유지되었으며, threshold 최적화를 통해 추가적인 FN/FP 균형 조정 여지가 있습니다.


> **학습 곡선, 평가 곡선 및 Grad-CAM 시각화:**

학습 이력:

![학습 이력](experiments/model_centric/figures/training_history.png)

평가 곡선:

![평가 곡선](experiments/model_centric/figures/evaluation_curves.png)

Grad-CAM (결함 위치 시각화):

![Grad-CAM](experiments/model_centric/figures/gradcam.png)

**Grad-CAM 분석 — FN 원인:**

- **FN #1 (contrast 부족):** 결함이 배경 텍스처와 대비가 낮아 육안으로도 모호한 수준. 모델의 관심도가 분산되어 결함 영역에 집중하지 못함.
- **FN #2 (레터박스 경계 오반응):** 결함이 희미하게 존재하나, 레터박스 패딩 경계의 대비선이 결함보다 강한 신호로 작용하여 탐지 실패.
- **공통:** TP 케이스 일부(#1)에서도 레터박스 경계 반응이 관찰됨. 복수 케이스에서 반복되는 점으로 보아 단순 노이즈가 아닌, 모델이 패딩 영역의 엉뚱한 특성을 학습했을 가능성이 높음. 레터박스 fill 값을 ImageNet 평균에 가까운 값으로 재조정하는 방향으로 추가 개선 여지가 있음.

---


## 개발 로드맵

- [x] 프로젝트 초기 설정 — 디렉토리 구조, 설정 파일, 의존성 관리
- [x] 탐색적 데이터 분석 — 클래스 분포 분석, 픽셀 통계 계산, 파트 단위 데이터 분할
- [x] 베이스라인 모델 — Simple CNN 구현, 초기 성능 기준선 수립 및 실패 케이스 분석 (test F1=0.2632, AUC=0.5535)
- [x] 데이터 중심 개선 — CLAHE/증강 강화, BCEWithLogitsLoss pos_weight 보정 (test F1=0.2642, AUC=0.6008, AP=0.3668)
- [x] 모델 중심 개선 — EfficientNet-B0 전이학습, 2단계 파인튜닝 (test F1=0.7000, AUC=0.9691, AP=0.8959)

---


## 라이센스

이 프로젝트의 코드는 MIT License를 따릅니다.

데이터셋 라이센스는 상단 [데이터셋 섹션](#데이터셋)을 참조하세요. KolektorSDD 데이터셋은 **CC BY-NC-SA 4.0** 라이센스를 따르며, 상업적 이용이 금지됩니다.
