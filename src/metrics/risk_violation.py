"""Risk violation metrics for CRC validity assessment."""

from typing import List, Tuple
import numpy as np
from scipy import stats
from loguru import logger


class RiskViolationMetrics:
    """Metrics for assessing risk control validity."""
    
    @staticmethod
    def compute_risk_gap(
        decisions: np.ndarray,
        correctness: np.ndarray,
        target_risk: float,
    ) -> float:
        """Compute risk gap: actual_risk - target_risk.
        
        Positive gap means risk exceeds target (violation).
        
        Args:
            decisions: Binary decisions (1=answer, 0=abstain)
            correctness: Correctness labels
            target_risk: Target risk level (alpha)
        
        Returns:
            Risk gap
        """
        answered_mask = (decisions == 1)
        
        if not np.any(answered_mask):
            return 0.0
        
        answered_correctness = correctness[answered_mask]
        actual_risk = 1.0 - np.mean(answered_correctness)
        
        gap = actual_risk - target_risk
        
        return float(gap)
    
    @staticmethod
    def is_risk_violated(
        decisions: np.ndarray,
        correctness: np.ndarray,
        target_risk: float,
    ) -> bool:
        """Check if risk constraint is violated.
        
        Args:
            decisions: Binary decisions
            correctness: Correctness labels
            target_risk: Target risk level
        
        Returns:
            True if actual risk > target risk
        """
        gap = RiskViolationMetrics.compute_risk_gap(
            decisions, correctness, target_risk
        )
        return gap > 0
    
    @staticmethod
    def compute_violation_rate_bootstrap(
        decisions: np.ndarray,
        correctness: np.ndarray,
        target_risk: float,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Compute risk violation rate via bootstrap.
        
        Args:
            decisions: Binary decisions
            correctness: Correctness labels
            target_risk: Target risk level
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
        
        Returns:
            Tuple of (violation_rate, mean_gap, std_gap)
        """
        np.random.seed(seed)
        
        n = len(decisions)
        violations = []
        gaps = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            boot_decisions = decisions[indices]
            boot_correctness = correctness[indices]
            
            # Compute gap
            gap = RiskViolationMetrics.compute_risk_gap(
                boot_decisions, boot_correctness, target_risk
            )
            
            gaps.append(gap)
            violations.append(gap > 0)
        
        violation_rate = np.mean(violations)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        return float(violation_rate), float(mean_gap), float(std_gap)
    
    @staticmethod
    def compute_confidence_interval(
        decisions: np.ndarray,
        correctness: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Compute confidence interval for selective risk.
        
        Args:
            decisions: Binary decisions
            correctness: Correctness labels
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
        
        Returns:
            Tuple of (mean_risk, lower_bound, upper_bound)
        """
        np.random.seed(seed)
        
        n = len(decisions)
        risks = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            boot_decisions = decisions[indices]
            boot_correctness = correctness[indices]
            
            # Compute risk
            answered_mask = (boot_decisions == 1)
            if np.any(answered_mask):
                answered_correctness = boot_correctness[answered_mask]
                risk = 1.0 - np.mean(answered_correctness)
            else:
                risk = 0.0
            
            risks.append(risk)
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_bound = np.percentile(risks, 100 * alpha / 2)
        upper_bound = np.percentile(risks, 100 * (1 - alpha / 2))
        mean_risk = np.mean(risks)
        
        return float(mean_risk), float(lower_bound), float(upper_bound)
    
    @staticmethod
    def compute_calibration_size_effect(
        decisions_list: List[np.ndarray],
        correctness_list: List[np.ndarray],
        calibration_sizes: List[int],
        target_risk: float,
    ) -> dict:
        """Analyze effect of calibration set size on risk violation.
        
        Args:
            decisions_list: List of decision arrays (one per calibration size)
            correctness_list: List of correctness arrays
            calibration_sizes: List of calibration set sizes
            target_risk: Target risk level
        
        Returns:
            Dictionary with results for each calibration size
        """
        results = {}
        
        for i, (decisions, correctness, size) in enumerate(
            zip(decisions_list, correctness_list, calibration_sizes)
        ):
            gap = RiskViolationMetrics.compute_risk_gap(
                decisions, correctness, target_risk
            )
            
            violation_rate, mean_gap, std_gap = (
                RiskViolationMetrics.compute_violation_rate_bootstrap(
                    decisions, correctness, target_risk
                )
            )
            
            results[size] = {
                'gap': gap,
                'violation_rate': violation_rate,
                'mean_gap': mean_gap,
                'std_gap': std_gap,
            }
        
        return results


def compute_risk_violation_metrics(
    decisions: np.ndarray,
    correctness: np.ndarray,
    target_risk: float,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute all risk violation metrics.
    
    Args:
        decisions: Binary decisions
        correctness: Correctness labels
        target_risk: Target risk level
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dictionary of metrics
    """
    # Basic gap
    gap = RiskViolationMetrics.compute_risk_gap(decisions, correctness, target_risk)
    is_violated = RiskViolationMetrics.is_risk_violated(decisions, correctness, target_risk)
    
    # Bootstrap analysis
    violation_rate, mean_gap, std_gap = (
        RiskViolationMetrics.compute_violation_rate_bootstrap(
            decisions, correctness, target_risk, n_bootstrap
        )
    )
    
    # Confidence interval
    mean_risk, lower_ci, upper_ci = (
        RiskViolationMetrics.compute_confidence_interval(
            decisions, correctness, n_bootstrap=n_bootstrap
        )
    )
    
    return {
        'risk_gap': gap,
        'is_violated': is_violated,
        'violation_rate_bootstrap': violation_rate,
        'mean_gap_bootstrap': mean_gap,
        'std_gap_bootstrap': std_gap,
        'mean_risk': mean_risk,
        'risk_ci_lower': lower_ci,
        'risk_ci_upper': upper_ci,
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 100
    confidences = np.random.rand(n)
    correctness = (confidences > 0.4).astype(float)
    correctness[np.random.rand(n) < 0.2] = 1 - correctness[np.random.rand(n) < 0.2]
    
    threshold = 0.6
    decisions = (confidences >= threshold).astype(int)
    
    target_risk = 0.05
    
    # Compute metrics
    metrics = compute_risk_violation_metrics(
        decisions, correctness, target_risk, n_bootstrap=1000
    )
    
    print(f"Risk Violation Metrics (target risk = {target_risk}):")
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")

