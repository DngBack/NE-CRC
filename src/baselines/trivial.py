"""Trivial baselines: always-answer and always-abstain."""

from typing import Optional
import numpy as np

from ..data import PredictionResult


class AlwaysAnswer:
    """Trivial baseline: always answer (never abstain)."""
    
    def predict(
        self,
        n: int,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Always answer.
        
        Args:
            n: Number of samples
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        results = []
        
        for sample_id, query in zip(sample_ids, queries):
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=1.0,
                decision="answer",
                decision_reason="Always answer (trivial baseline)",
            )
            results.append(result)
        
        return results


class AlwaysAbstain:
    """Trivial baseline: always abstain (never answer)."""
    
    def predict(
        self,
        n: int,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Always abstain.
        
        Args:
            n: Number of samples
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        results = []
        
        for sample_id, query in zip(sample_ids, queries):
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=0.0,
                decision="abstain",
                decision_reason="Always abstain (trivial baseline)",
            )
            results.append(result)
        
        return results


def create_always_answer() -> AlwaysAnswer:
    """Create always-answer baseline."""
    return AlwaysAnswer()


def create_always_abstain() -> AlwaysAbstain:
    """Create always-abstain baseline."""
    return AlwaysAbstain()


if __name__ == "__main__":
    # Example usage
    n = 10
    
    # Always answer
    always_answer = create_always_answer()
    results_answer = always_answer.predict(n)
    
    print("Always Answer:")
    answered = sum(1 for r in results_answer if r.decision == "answer")
    print(f"  Answered: {answered}/{len(results_answer)}")
    
    # Always abstain
    always_abstain = create_always_abstain()
    results_abstain = always_abstain.predict(n)
    
    print("\nAlways Abstain:")
    answered = sum(1 for r in results_abstain if r.decision == "answer")
    print(f"  Answered: {answered}/{len(results_abstain)}")

