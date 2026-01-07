"""Cost metrics for system efficiency evaluation."""

from typing import Dict, List
import numpy as np
from loguru import logger


class CostMetrics:
    """Compute cost-related metrics."""
    
    @staticmethod
    def compute_total_tokens(
        token_counts: np.ndarray,
    ) -> int:
        """Compute total tokens used.
        
        Args:
            token_counts: Array of token counts per sample
        
        Returns:
            Total tokens
        """
        return int(np.sum(token_counts))
    
    @staticmethod
    def compute_mean_tokens_per_sample(
        token_counts: np.ndarray,
    ) -> float:
        """Compute mean tokens per sample.
        
        Args:
            token_counts: Array of token counts per sample
        
        Returns:
            Mean tokens per sample
        """
        return float(np.mean(token_counts))
    
    @staticmethod
    def compute_tokens_per_answered(
        token_counts: np.ndarray,
        decisions: np.ndarray,
    ) -> float:
        """Compute average tokens per answered sample.
        
        Args:
            token_counts: Array of token counts per sample
            decisions: Binary decisions (1=answer, 0=abstain)
        
        Returns:
            Mean tokens per answered sample
        """
        answered_mask = (decisions == 1)
        
        if not np.any(answered_mask):
            return 0.0
        
        answered_tokens = token_counts[answered_mask]
        return float(np.mean(answered_tokens))
    
    @staticmethod
    def compute_latency_metrics(
        latencies: np.ndarray,
    ) -> Dict[str, float]:
        """Compute latency statistics.
        
        Args:
            latencies: Array of latencies (ms) per sample
        
        Returns:
            Dictionary of latency metrics
        """
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'total_latency_ms': float(np.sum(latencies)),
        }
    
    @staticmethod
    def compute_cost_risk_tradeoff(
        token_counts: np.ndarray,
        decisions: np.ndarray,
        correctness: np.ndarray,
    ) -> Dict[str, float]:
        """Compute cost-risk trade-off metrics.
        
        Args:
            token_counts: Token counts per sample
            decisions: Binary decisions
            correctness: Correctness labels
        
        Returns:
            Dictionary of trade-off metrics
        """
        # Total tokens
        total_tokens = CostMetrics.compute_total_tokens(token_counts)
        
        # Coverage and risk
        answered_mask = (decisions == 1)
        coverage = np.mean(answered_mask)
        
        if np.any(answered_mask):
            answered_correctness = correctness[answered_mask]
            selective_risk = 1.0 - np.mean(answered_correctness)
        else:
            selective_risk = 0.0
        
        # Cost efficiency metrics
        if coverage > 0:
            cost_per_coverage = total_tokens / coverage
        else:
            cost_per_coverage = float('inf')
        
        if selective_risk > 0:
            cost_per_risk = total_tokens / selective_risk
        else:
            cost_per_risk = float('inf')
        
        # Quality per token
        if total_tokens > 0:
            coverage_per_1k_tokens = (coverage * 1000) / total_tokens
            accuracy = 1.0 - selective_risk
            accuracy_per_1k_tokens = (accuracy * 1000) / total_tokens if np.any(answered_mask) else 0
        else:
            coverage_per_1k_tokens = 0.0
            accuracy_per_1k_tokens = 0.0
        
        return {
            'total_tokens': total_tokens,
            'coverage': float(coverage),
            'selective_risk': float(selective_risk),
            'cost_per_coverage': float(cost_per_coverage),
            'coverage_per_1k_tokens': float(coverage_per_1k_tokens),
            'accuracy_per_1k_tokens': float(accuracy_per_1k_tokens),
        }
    
    @staticmethod
    def compute_efficiency_score(
        token_counts: np.ndarray,
        decisions: np.ndarray,
        correctness: np.ndarray,
        coverage_weight: float = 0.5,
        accuracy_weight: float = 0.5,
    ) -> float:
        """Compute overall efficiency score.
        
        Score = (coverage_weight * coverage + accuracy_weight * accuracy) / (tokens / 1000)
        
        Higher is better (more quality per token).
        
        Args:
            token_counts: Token counts per sample
            decisions: Binary decisions
            correctness: Correctness labels
            coverage_weight: Weight for coverage
            accuracy_weight: Weight for accuracy
        
        Returns:
            Efficiency score
        """
        total_tokens = CostMetrics.compute_total_tokens(token_counts)
        
        if total_tokens == 0:
            return 0.0
        
        answered_mask = (decisions == 1)
        coverage = np.mean(answered_mask)
        
        if np.any(answered_mask):
            answered_correctness = correctness[answered_mask]
            accuracy = np.mean(answered_correctness)
        else:
            accuracy = 0.0
        
        # Normalize by 1k tokens
        tokens_1k = total_tokens / 1000.0
        
        # Weighted quality per 1k tokens
        quality = coverage_weight * coverage + accuracy_weight * accuracy
        efficiency = quality / tokens_1k if tokens_1k > 0 else 0.0
        
        return float(efficiency)


def compute_cost_metrics(
    token_counts: np.ndarray,
    decisions: np.ndarray,
    correctness: np.ndarray,
    latencies: np.ndarray = None,
) -> dict:
    """Compute all cost-related metrics.
    
    Args:
        token_counts: Token counts per sample
        decisions: Binary decisions
        correctness: Correctness labels
        latencies: Optional latency measurements
    
    Returns:
        Dictionary of cost metrics
    """
    metrics = {}
    
    # Token metrics
    metrics['total_tokens'] = CostMetrics.compute_total_tokens(token_counts)
    metrics['mean_tokens_per_sample'] = CostMetrics.compute_mean_tokens_per_sample(token_counts)
    metrics['tokens_per_answered'] = CostMetrics.compute_tokens_per_answered(
        token_counts, decisions
    )
    
    # Latency metrics
    if latencies is not None:
        latency_metrics = CostMetrics.compute_latency_metrics(latencies)
        metrics.update(latency_metrics)
    
    # Cost-risk trade-off
    tradeoff_metrics = CostMetrics.compute_cost_risk_tradeoff(
        token_counts, decisions, correctness
    )
    metrics.update(tradeoff_metrics)
    
    # Efficiency score
    metrics['efficiency_score'] = CostMetrics.compute_efficiency_score(
        token_counts, decisions, correctness
    )
    
    return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 100
    token_counts = np.random.randint(50, 500, size=n)
    latencies = np.random.uniform(100, 1000, size=n)
    
    confidences = np.random.rand(n)
    correctness = (confidences > 0.5).astype(float)
    correctness[np.random.rand(n) < 0.2] = 1 - correctness[np.random.rand(n) < 0.2]
    
    threshold = 0.6
    decisions = (confidences >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_cost_metrics(token_counts, decisions, correctness, latencies)
    
    print("Cost Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

