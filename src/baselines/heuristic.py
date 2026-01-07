"""Heuristic threshold baseline (direct uncertainty threshold)."""

from typing import Optional
import numpy as np
from loguru import logger

from ..data import PredictionResult


class HeuristicThreshold:
    """Simple heuristic: threshold directly on uncertainty without calibration."""
    
    def __init__(
        self,
        threshold: float = 0.5,
        invert: bool = False,
    ):
        """Initialize heuristic threshold.
        
        Args:
            threshold: Uncertainty threshold
            invert: If True, abstain when uncertainty > threshold
                   If False, abstain when uncertainty < threshold
        """
        self.threshold = threshold
        self.invert = invert
    
    def predict(
        self,
        uncertainties: np.ndarray,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Make predictions based on uncertainty threshold.
        
        Args:
            uncertainties: Uncertainty values
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        n = len(uncertainties)
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        results = []
        
        for i, (uncertainty, sample_id, query) in enumerate(
            zip(uncertainties, sample_ids, queries)
        ):
            # Decision logic
            if self.invert:
                # Low uncertainty = confident = answer
                should_answer = (uncertainty <= self.threshold)
            else:
                # High uncertainty = uncertain = abstain
                should_answer = (uncertainty >= self.threshold)
            
            decision = "answer" if should_answer else "abstain"
            reason = f"Uncertainty {uncertainty:.3f} {'<=' if self.invert else '>='} threshold {self.threshold}"
            
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                uncertainty=uncertainty,
                decision=decision,
                decision_reason=reason,
                threshold=self.threshold,
            )
            
            results.append(result)
        
        return results


def create_heuristic_baseline(
    threshold: float = 0.5,
    invert: bool = True,
) -> HeuristicThreshold:
    """Factory function to create heuristic baseline.
    
    Args:
        threshold: Uncertainty threshold
        invert: Whether to invert logic (True = abstain when high uncertainty)
    
    Returns:
        HeuristicThreshold instance
    """
    return HeuristicThreshold(threshold=threshold, invert=invert)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 20
    uncertainties = np.random.rand(n) * 2  # 0 to 2
    
    # Create baseline
    baseline = create_heuristic_baseline(threshold=1.0, invert=True)
    results = baseline.predict(uncertainties)
    
    print("Heuristic Threshold Baseline:")
    for i, result in enumerate(results[:10]):
        print(f"  Sample {i}: u={result.uncertainty:.3f} -> {result.decision}")
    
    answered = sum(1 for r in results if r.decision == "answer")
    print(f"\nAnswered: {answered}/{len(results)}")

