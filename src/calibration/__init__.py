"""Calibration heads module for UniCR."""

from .logistic_head import (
    LogisticCalibrationHead,
    create_logistic_head,
)
from .mlp_head import (
    MLPCalibrationHead,
    create_mlp_head,
)

__all__ = [
    # Logistic regression
    "LogisticCalibrationHead",
    "create_logistic_head",
    # MLP
    "MLPCalibrationHead",
    "create_mlp_head",
]

