# KolektorSDD 표면 결함 분류 (Surface Defect Classification)

> KolektorSDD 산업용 표면 결함 데이터셋을 활용한 머신비전 이진 분류 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 목차 (Table of Contents)

1. [프로젝트 개요](#프로젝트-개요-overview)
2. [데이터셋](#데이터셋-dataset)
3. [프로젝트 구조](#프로젝트-구조-structure)
4. [설치 및 실행](#설치-및-실행-setup--run)
5. [결과](#결과-results)
6. [개발 로드맵](#개발-로드맵-roadmap)
7. [라이센스](#라이센스-license)

---

## 프로젝트 개요 (Overview)

KolektorSDD 산업용 표면 결함 데이터셋을 활용한 이진 분류 (Binary Classification) 프로젝트입니다. 총 399장의 이미지 중 52장(결함)과 347장(정상)으로 구성된 심한 클래스 불균형 (Class Imbalance) 환경에서, 데이터 중심 개선 (Data-Centric) 과 모델 중심 개선 (Model-Centric) 을 결합하여 결함 탐지 성능을 높이는 것이 핵심 목적입니다.

---

## 데이터셋 (Dataset)

### KolektorSDD

| 항목 | 내용 |
|---|---|
| 전체 이미지 | 399장 |
| 결함 (Defect) | 52장 |
| 정상 (Normal) | 347장 |
| 불균형 비율 (Imbalance Ratio) | ~1:6.7 (결함:정상) |
| 파트 수 (Parts) | 50개 |
| 이미지 크기 범위 | 500x1235 ~ 500x1285 픽셀 |

**출처 및 저작권:**
- 논문: "Segmentation-based deep-learning approach for surface-defect detection"
- 데이터셋 공식 페이지: https://www.vicos.si/resources/kolektorsdd/
- 라이센스: **CC BY-NC-SA 4.0**

**Citation (BibTeX):**
```bibtex
@article{tabernik2020segmentation,
  title={Segmentation-based deep-learning approach for surface-defect detection},
  author={Tabernik, Domen and {\v{S}}ela, Samo and Skvar{\v{c}}, Jure and Sko{\v{c}}aj, Danijel},
  journal={Journal of Intelligent Manufacturing},
  volume={31}, number={3}, pages={759--776}, year={2020}
}
```

**사용 안내:**
- 본 프로젝트는 **교육 및 포트폴리오 목적**으로 데이터셋을 사용합니다.
- 원본 데이터 파일은 저장소에 포함되지 않습니다 (`.gitignore` 처리).
- 데이터셋은 위 공식 출처에서 직접 다운로드하시기 바랍니다.

---

## 프로젝트 구조 (Structure)

```
kolektorsdd-classification/
├── configs/
│   └── base.yaml                    # 공통 설정 (Config)
├── data/
│   ├── raw/                         # 원본 데이터 (gitignore)
│   └── processed/
│       ├── train.csv                # 학습 분할 (Split)
│       ├── val.csv                  # 검증 분할
│       └── test.csv                 # 테스트 분할
├── experiments/                     # 실험 결과 (학습 후 생성)
├── notebooks/
│   └── 01_eda.ipynb                 # 탐색적 데이터 분석 노트북
├── reports/
│   └── figures/                     # EDA 시각화 이미지
├── src/
│   ├── data/                        # 데이터셋 및 전처리
│   ├── models/                      # 모델 정의
│   ├── training/                    # 학습 루프
│   ├── evaluation/                  # 평가 지표
│   └── utils/                       # 유틸리티
├── requirements.txt
└── pyproject.toml
```

---

## 설치 및 실행 (Setup & Run)

### 1. 환경 설정

```bash
git clone <repository-url>
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

## 결과 (Results)

### 탐색적 데이터 분석 (Exploratory Data Analysis)

데이터셋 전반을 분석하고, 파트 단위 계층적 분할 (Part-level Stratified Split) 을 완료했습니다. 동일 파트의 이미지가 학습/검증/테스트에 중복 포함되지 않도록 하여 데이터 누수 (Data Leakage) 를 방지했습니다.

**픽셀 통계 (train 기준):**

| 항목 | 값 |
|---|---|
| 평균 (Mean) | 0.7317 |
| 표준편차 (Std) | 0.0915 |
| 이미지 크기 범위 | 500x1235 ~ 500x1285 |

**데이터 분할 결과 (Part-level Stratified Split):**

| 분할 (Split) | 전체 | 결함 (Defect) | 정상 (Normal) |
|---|---|---|---|
| train | 280장 | 36장 | 244장 |
| val | 56장 | 7장 | 49장 |
| test | 63장 | 9장 | 54장 |

**EDA 시각화:**

클래스 분포 (Class Distribution):

![클래스 분포](reports/figures/class_distribution.png)

정상 샘플 (Normal Samples):

![정상 샘플](reports/figures/normal_samples.png)

결함 샘플 (Defect Samples):

![결함 샘플](reports/figures/defect_samples.png)

픽셀 강도 분포 (Pixel Intensity Distribution):

![픽셀 히스토그램](reports/figures/pixel_histogram.png)

---

## 개발 로드맵 (Roadmap)

- [x] 프로젝트 초기 설정 — 디렉토리 구조, 설정 파일, 의존성 관리
- [x] 탐색적 데이터 분석 — 클래스 분포 분석, 픽셀 통계 계산, 파트 단위 데이터 분할
- 이후 단계 진행 예정

---

## 라이센스 (License)

이 프로젝트의 **코드**는 MIT License를 따릅니다.

데이터셋 라이센스는 상단 [데이터셋 섹션](#데이터셋-dataset)을 참조하세요. KolektorSDD 데이터셋은 **CC BY-NC-SA 4.0** 라이센스를 따르며, 상업적 이용이 금지됩니다.
