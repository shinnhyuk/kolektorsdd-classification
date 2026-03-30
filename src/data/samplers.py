"""WeightedRandomSampler 빌더.

극단적 클래스 불균형(train: 36 defect / 244 normal = 6.78:1)을 처리하기 위해
각 샘플에 클래스 빈도의 역수를 가중치로 부여하여 결함 샘플을 과표집한다.

가중치 계산 원칙:
    weight[i] = 1.0 / class_count[label[i]]
    → 소수 클래스(결함)와 다수 클래스(정상)가 배치 내에서 균등하게 추출됨
"""

from collections import Counter
from typing import List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """클래스 빈도 기반 WeightedRandomSampler를 생성하여 반환한다.

    dataset.get_labels()로 레이블 목록을 취득한 뒤 클래스별 빈도를 계산하고,
    각 샘플의 가중치를 해당 클래스 빈도의 역수로 설정한다.
    이를 통해 배치 내 클래스 비율이 자동으로 균등화된다.

    Args:
        dataset: get_labels() 메서드를 구현한 PyTorch Dataset.
                 KolektorDataset이 이 인터페이스를 충족한다.

    Returns:
        WeightedRandomSampler — DataLoader의 sampler 인자로 바로 사용 가능.
        num_samples=len(dataset), replacement=True로 설정된다.

    Example:
        >>> sampler = build_weighted_sampler(train_ds)
        >>> loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    """
    labels: List[int] = dataset.get_labels()

    # 클래스별 샘플 수 계산
    class_counts: Counter = Counter(labels)

    # 각 샘플에 클래스 빈도의 역수를 가중치로 할당
    # 예) defect(1) count=36 → weight=1/36, normal(0) count=244 → weight=1/244
    weights: List[float] = [1.0 / class_counts[int(label)] for label in labels]
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(dataset),
        replacement=True,
    )

    return sampler
