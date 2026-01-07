"""Metrics module for evaluation."""

from .coverage_risk import (
    CoverageRiskMetrics,
    compute_coverage_risk_metrics,
)
from .rc_curve import (
    RCCurve,
    compute_rc_metrics,
)
from .risk_violation import (
    RiskViolationMetrics,
    compute_risk_violation_metrics,
)
from .cost_metrics import (
    CostMetrics,
    compute_cost_metrics,
)
from .metrics_computer import (
    MetricsComputer,
    create_metrics_computer,
)

__all__ = [
    # Coverage-Risk
    "CoverageRiskMetrics",
    "compute_coverage_risk_metrics",
    # RC Curve
    "RCCurve",
    "compute_rc_metrics",
    # Risk Violation
    "RiskViolationMetrics",
    "compute_risk_violation_metrics",
    # Cost
    "CostMetrics",
    "compute_cost_metrics",
    # Unified
    "MetricsComputer",
    "create_metrics_computer",
]

