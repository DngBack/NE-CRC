"""Unified metrics computation interface."""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from .coverage_risk import compute_coverage_risk_metrics
from .rc_curve import compute_rc_metrics
from .risk_violation import compute_risk_violation_metrics
from .cost_metrics import compute_cost_metrics


class MetricsComputer:
    """Unified interface for computing all evaluation metrics."""
    
    def __init__(
        self,
        target_risks: List[float] = [0.01, 0.02, 0.05, 0.10],
        n_bootstrap: int = 1000,
        rc_num_points: int = 100,
    ):
        """Initialize metrics computer.
        
        Args:
            target_risks: List of target risk levels for coverage@risk
            n_bootstrap: Number of bootstrap samples for CI
            rc_num_points: Number of points for RC curve
        """
        self.target_risks = target_risks
        self.n_bootstrap = n_bootstrap
        self.rc_num_points = rc_num_points
    
    def compute_all(
        self,
        decisions: np.ndarray,
        confidences: np.ndarray,
        correctness: np.ndarray,
        token_counts: Optional[np.ndarray] = None,
        latencies: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ) -> Dict:
        """Compute all metrics.
        
        Args:
            decisions: Binary decisions (1=answer, 0=abstain)
            confidences: Confidence scores
            correctness: Correctness labels (1=correct, 0=incorrect)
            token_counts: Optional token counts per sample
            latencies: Optional latency measurements (ms)
            alpha: Target risk level for violation metrics
        
        Returns:
            Dictionary of all metrics
        """
        logger.info("Computing all evaluation metrics")
        
        metrics = {}
        
        # Coverage-Risk metrics
        cr_metrics = compute_coverage_risk_metrics(
            decisions, confidences, correctness, target_risks=self.target_risks
        )
        metrics.update(cr_metrics)
        
        # RC curve and AURC
        rc_metrics = compute_rc_metrics(
            confidences, correctness, num_points=self.rc_num_points
        )
        # Don't include curve arrays in main metrics dict
        metrics['aurc'] = rc_metrics['aurc']
        metrics['normalized_aurc'] = rc_metrics['normalized_aurc']
        metrics['eaurc'] = rc_metrics['eaurc']
        # Store curves separately for plotting
        metrics['_rc_curve'] = {
            'coverages': rc_metrics['coverages'],
            'risks': rc_metrics['risks'],
        }
        
        # Risk violation metrics
        violation_metrics = compute_risk_violation_metrics(
            decisions, correctness, alpha, n_bootstrap=self.n_bootstrap
        )
        metrics.update(violation_metrics)
        
        # Cost metrics (if available)
        if token_counts is not None:
            cost_metrics = compute_cost_metrics(
                token_counts, decisions, correctness, latencies
            )
            metrics.update(cost_metrics)
        
        logger.info(f"Computed {len(metrics)} metrics")
        
        return metrics
    
    def compute_per_scenario(
        self,
        decisions_dict: Dict[str, np.ndarray],
        confidences_dict: Dict[str, np.ndarray],
        correctness_dict: Dict[str, np.ndarray],
        token_counts_dict: Optional[Dict[str, np.ndarray]] = None,
        latencies_dict: Optional[Dict[str, np.ndarray]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Dict]:
        """Compute metrics per scenario.
        
        Args:
            decisions_dict: Dictionary mapping scenario names to decisions
            confidences_dict: Dictionary mapping scenario names to confidences
            correctness_dict: Dictionary mapping scenario names to correctness
            token_counts_dict: Optional token counts per scenario
            latencies_dict: Optional latencies per scenario
            alpha: Target risk level
        
        Returns:
            Dictionary mapping scenario names to metrics
        """
        logger.info(f"Computing metrics for {len(decisions_dict)} scenarios")
        
        results = {}
        
        for scenario in decisions_dict.keys():
            logger.info(f"Computing metrics for scenario: {scenario}")
            
            token_counts = token_counts_dict.get(scenario) if token_counts_dict else None
            latencies = latencies_dict.get(scenario) if latencies_dict else None
            
            metrics = self.compute_all(
                decisions_dict[scenario],
                confidences_dict[scenario],
                correctness_dict[scenario],
                token_counts,
                latencies,
                alpha,
            )
            
            results[scenario] = metrics
        
        return results
    
    def compute_aggregate(
        self,
        per_scenario_metrics: Dict[str, Dict],
        aggregation: str = "macro",
    ) -> Dict:
        """Aggregate metrics across scenarios.
        
        Args:
            per_scenario_metrics: Dictionary of metrics per scenario
            aggregation: Aggregation method ("macro" or "weighted")
        
        Returns:
            Aggregated metrics
        """
        logger.info(f"Aggregating metrics with {aggregation} averaging")
        
        if not per_scenario_metrics:
            return {}
        
        # Get all metric keys (excluding non-numeric)
        first_scenario = list(per_scenario_metrics.values())[0]
        metric_keys = [
            k for k, v in first_scenario.items()
            if isinstance(v, (int, float, np.number)) and not k.startswith('_')
        ]
        
        aggregate = {}
        
        if aggregation == "macro":
            # Simple average across scenarios
            for key in metric_keys:
                values = [
                    metrics[key] for metrics in per_scenario_metrics.values()
                    if key in metrics
                ]
                if values:
                    aggregate[f'macro_{key}'] = float(np.mean(values))
                    aggregate[f'std_{key}'] = float(np.std(values))
        
        elif aggregation == "weighted":
            # Weight by number of samples per scenario
            num_samples_key = 'num_total'
            
            if num_samples_key in first_scenario:
                for key in metric_keys:
                    if key == num_samples_key:
                        continue
                    
                    weighted_sum = 0.0
                    total_weight = 0.0
                    
                    for scenario, metrics in per_scenario_metrics.items():
                        if key in metrics and num_samples_key in metrics:
                            weight = metrics[num_samples_key]
                            weighted_sum += metrics[key] * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        aggregate[f'weighted_{key}'] = weighted_sum / total_weight
        
        return aggregate
    
    def format_metrics_table(
        self,
        metrics: Dict,
        decimal_places: int = 3,
    ) -> str:
        """Format metrics as a readable table.
        
        Args:
            metrics: Dictionary of metrics
            decimal_places: Number of decimal places
        
        Returns:
            Formatted table string
        """
        lines = ["Metrics:"]
        lines.append("-" * 60)
        
        for key, value in sorted(metrics.items()):
            if key.startswith('_'):
                continue  # Skip internal data
            
            if isinstance(value, bool):
                lines.append(f"  {key:40s}: {value}")
            elif isinstance(value, (int, np.integer)):
                lines.append(f"  {key:40s}: {value:,}")
            elif isinstance(value, (float, np.floating)):
                lines.append(f"  {key:40s}: {value:.{decimal_places}f}")
        
        return "\n".join(lines)


def create_metrics_computer(**kwargs) -> MetricsComputer:
    """Factory function to create metrics computer.
    
    Args:
        **kwargs: Arguments for MetricsComputer
    
    Returns:
        MetricsComputer instance
    """
    return MetricsComputer(**kwargs)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 200
    confidences = np.random.rand(n)
    correctness = (confidences > 0.5).astype(float)
    correctness[np.random.rand(n) < 0.2] = 1 - correctness[np.random.rand(n) < 0.2]
    
    threshold = 0.6
    decisions = (confidences >= threshold).astype(int)
    
    token_counts = np.random.randint(50, 500, size=n)
    latencies = np.random.uniform(100, 1000, size=n)
    
    # Compute all metrics
    computer = create_metrics_computer()
    metrics = computer.compute_all(
        decisions, confidences, correctness, token_counts, latencies, alpha=0.05
    )
    
    # Print formatted table
    table = computer.format_metrics_table(metrics)
    print(table)
    
    # Test per-scenario computation
    scenarios = {
        'scenario_A': decisions[:100],
        'scenario_B': decisions[100:],
    }
    confidences_dict = {
        'scenario_A': confidences[:100],
        'scenario_B': confidences[100:],
    }
    correctness_dict = {
        'scenario_A': correctness[:100],
        'scenario_B': correctness[100:],
    }
    
    per_scenario = computer.compute_per_scenario(
        scenarios, confidences_dict, correctness_dict
    )
    
    print("\n\nPer-scenario results:")
    for scenario, metrics in per_scenario.items():
        print(f"\n{scenario}:")
        print(f"  Coverage: {metrics['coverage']:.3f}")
        print(f"  Selective risk: {metrics['selective_risk']:.3f}")
        print(f"  AURC: {metrics['aurc']:.4f}")
    
    # Aggregate
    aggregate = computer.compute_aggregate(per_scenario)
    print("\n\nAggregate metrics:")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.4f}")

