"""Coverage and selective risk metrics."""

from typing import Tuple, Optional
import numpy as np
from loguru import logger


class CoverageRiskMetrics:
    """Compute coverage and selective risk metrics."""
    
    @staticmethod
    def compute_coverage(
        decisions: np.ndarray,
    ) -> float:
        """Compute coverage (fraction of samples answered).
        
        Args:
            decisions: Binary decisions (1=answer, 0=abstain)
        
        Returns:
            Coverage in [0, 1]
        """
        return float(np.mean(decisions))
    
    @staticmethod
    def compute_selective_risk(
        decisions: np.ndarray,
        correctness: np.ndarray,
    ) -> float:
        """Compute selective risk (error rate on answered samples).
        
        Args:
            decisions: Binary decisions (1=answer, 0=abstain)
            correctness: Correctness labels (1=correct, 0=incorrect)
        
        Returns:
            Selective risk in [0, 1] (or 0.0 if no samples answered)
        """
        answered_mask = (decisions == 1)
        
        if not np.any(answered_mask):
            return 0.0
        
        answered_correctness = correctness[answered_mask]
        risk = 1.0 - np.mean(answered_correctness)
        
        return float(risk)
    
    @staticmethod
    def compute_coverage_at_risk(
        confidences: np.ndarray,
        correctness: np.ndarray,
        target_risk: float,
    ) -> Tuple[float, float]:
        """Compute maximum coverage achievable at target risk level.
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels
            target_risk: Target risk level (e.g., 0.05 for 5% error)
        
        Returns:
            Tuple of (coverage, actual_risk)
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_correctness = correctness[sorted_indices]
        
        n = len(confidences)
        best_coverage = 0.0
        best_risk = 0.0
        
        # Try different thresholds (by including top-k samples)
        for k in range(1, n + 1):
            top_k_correctness = sorted_correctness[:k]
            risk = 1.0 - np.mean(top_k_correctness)
            coverage = k / n
            
            if risk <= target_risk:
                best_coverage = coverage
                best_risk = risk
        
        return best_coverage, best_risk
    
    @staticmethod
    def compute_risk_at_coverage(
        confidences: np.ndarray,
        correctness: np.ndarray,
        target_coverage: float,
    ) -> float:
        """Compute risk at target coverage level.
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels
            target_coverage: Target coverage (e.g., 0.8 for 80%)
        
        Returns:
            Risk at target coverage
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_correctness = correctness[sorted_indices]
        
        n = len(confidences)
        k = int(n * target_coverage)
        
        if k == 0:
            return 0.0
        
        top_k_correctness = sorted_correctness[:k]
        risk = 1.0 - np.mean(top_k_correctness)
        
        return float(risk)
    
    @staticmethod
    def compute_metrics(
        decisions: np.ndarray,
        confidences: np.ndarray,
        correctness: np.ndarray,
        target_risks: list = [0.01, 0.02, 0.05, 0.10],
    ) -> dict:
        """Compute all coverage-risk metrics.
        
        Args:
            decisions: Binary decisions
            confidences: Confidence scores
            correctness: Correctness labels
            target_risks: List of target risk levels
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['coverage'] = CoverageRiskMetrics.compute_coverage(decisions)
        metrics['selective_risk'] = CoverageRiskMetrics.compute_selective_risk(
            decisions, correctness
        )
        
        # Coverage at different risk levels
        for risk in target_risks:
            cov, actual_risk = CoverageRiskMetrics.compute_coverage_at_risk(
                confidences, correctness, risk
            )
            metrics[f'coverage@risk<={risk:.2f}'] = cov
            metrics[f'actual_risk@coverage_{cov:.2f}'] = actual_risk
        
        # Number of samples
        metrics['num_answered'] = int(np.sum(decisions))
        metrics['num_total'] = len(decisions)
        
        return metrics


def compute_coverage_risk_metrics(
    decisions: np.ndarray,
    confidences: np.ndarray,
    correctness: np.ndarray,
    **kwargs,
) -> dict:
    """Convenience function to compute all coverage-risk metrics.
    
    Args:
        decisions: Binary decisions
        confidences: Confidence scores
        correctness: Correctness labels
        **kwargs: Additional arguments
    
    Returns:
        Dictionary of metrics
    """
    return CoverageRiskMetrics.compute_metrics(decisions, confidences, correctness, **kwargs)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 100
    confidences = np.random.rand(n)
    correctness = (confidences > 0.5).astype(float)
    correctness[np.random.rand(n) < 0.2] = 1 - correctness[np.random.rand(n) < 0.2]
    
    # Decisions based on threshold
    threshold = 0.6
    decisions = (confidences >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_coverage_risk_metrics(decisions, confidences, correctness)
    
    print("Coverage-Risk Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

