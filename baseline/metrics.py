"""Competition metrics and class weights for SCNU table relation extraction."""

from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def compute_class_weights(label_counts: pd.Series) -> Dict[str, float]:
    """Few-shot importance weights from official competition formula."""
    if label_counts.empty:
        raise ValueError('label_counts is empty')

    counts = label_counts.astype(float)
    counts_max = float(counts.max())
    counts_min = float(counts.min())
    denom = counts_max + counts_min * 0.1
    if denom <= 0:
        raise ValueError('invalid counts for weight computation')

    weights = {}
    for label, count in counts.items():
        weights[str(label)] = (counts_max - float(count) + counts_min * 0.1) / denom
    return weights


def normalize_weights(label_to_weight: Dict[str, float]) -> Dict[str, float]:
    values = list(label_to_weight.values())
    mean_weight = float(np.mean(values)) if values else 1.0
    if mean_weight <= 0:
        return label_to_weight
    return {label: weight / mean_weight for label, weight in label_to_weight.items()}


def weights_tensor_for_encoder(
    label_encoder: LabelEncoder,
    label_to_weight: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    weights = []
    for label in label_encoder.classes_:
        if label not in label_to_weight:
            raise KeyError(f'missing weight for label: {label}')
        weights.append(float(label_to_weight[label]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_accuracy(
    y_true: Sequence[Union[int, str]],
    y_pred: Sequence[Union[int, str]],
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f'length mismatch: {len(y_true)} vs {len(y_pred)}')
    if not y_true:
        return 0.0
    return float(np.mean([gt == pred for gt, pred in zip(y_true, y_pred)]))


def compute_score_final(
    y_true: Sequence[Union[int, str]],
    y_pred: Sequence[Union[int, str]],
    label_to_weight: Dict[str, float],
) -> float:
    """Weighted per-class accuracy (Score_final) for labels present in y_true."""
    if len(y_true) != len(y_pred):
        raise ValueError(f'length mismatch: {len(y_true)} vs {len(y_pred)}')
    if not y_true:
        return 0.0

    y_true_list = [str(item) for item in y_true]
    y_pred_list = [str(item) for item in y_pred]

    weighted_score = 0.0
    weight_sum = 0.0
    for label in sorted(set(y_true_list)):
        indices = [idx for idx, value in enumerate(y_true_list) if value == label]
        m_total = len(indices)
        if m_total == 0:
            continue
        m_correct = sum(1 for idx in indices if y_pred_list[idx] == label)
        m_score = m_correct / m_total
        m_weight = float(label_to_weight.get(label, 0.0))
        weighted_score += m_weight * m_score
        weight_sum += m_weight

    return weighted_score / weight_sum if weight_sum > 0 else 0.0


def ids_to_labels(label_encoder: LabelEncoder, label_ids: Iterable[int]) -> List[str]:
    return [str(label_encoder.classes_[int(label_id)]) for label_id in label_ids]
