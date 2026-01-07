"""Unified interface for uncertainty estimation."""

from typing import List, Dict, Any
from loguru import logger

from .semantic_entropy import SemanticEntropyEstimator, create_semantic_entropy_estimator
from .token_entropy import TokenEntropyEstimator, create_token_entropy_estimator
from .dispersion import DispersionEstimator, create_dispersion_estimator


class UncertaintyEstimator:
    """Unified interface for computing various uncertainty measures."""
    
    def __init__(
        self,
        methods: List[str] = ["semantic_entropy", "token_entropy", "dispersion"],
        semantic_entropy_config: Dict[str, Any] = None,
        token_entropy_config: Dict[str, Any] = None,
        dispersion_config: Dict[str, Any] = None,
    ):
        """Initialize uncertainty estimator with multiple methods.
        
        Args:
            methods: List of uncertainty methods to use
            semantic_entropy_config: Config for semantic entropy
            token_entropy_config: Config for token entropy
            dispersion_config: Config for dispersion
        """
        self.methods = methods
        
        # Initialize estimators
        self.estimators = {}
        
        if "semantic_entropy" in methods:
            config = semantic_entropy_config or {}
            self.estimators["semantic_entropy"] = create_semantic_entropy_estimator(**config)
            logger.info("Initialized semantic entropy estimator")
        
        if "token_entropy" in methods:
            config = token_entropy_config or {}
            self.estimators["token_entropy"] = create_token_entropy_estimator(**config)
            logger.info("Initialized token entropy estimator")
        
        if "dispersion" in methods:
            config = dispersion_config or {}
            self.estimators["dispersion"] = create_dispersion_estimator(**config)
            logger.info("Initialized dispersion estimator")
    
    def compute(
        self,
        generations: List[str],
        token_logprobs: List[List[float]] = None,
        method: str = None,
    ) -> float:
        """Compute uncertainty using specified method.
        
        Args:
            generations: List of generated texts
            token_logprobs: Token log probabilities (required for token_entropy)
            method: Specific method to use (or None for primary method)
        
        Returns:
            Uncertainty value
        """
        if method is None:
            method = self.methods[0]  # Use first method as default
        
        if method not in self.estimators:
            raise ValueError(
                f"Method {method} not initialized. "
                f"Available: {list(self.estimators.keys())}"
            )
        
        if method == "semantic_entropy":
            return self.estimators["semantic_entropy"].compute(generations)
        
        elif method == "token_entropy":
            if token_logprobs is None:
                logger.warning("Token logprobs not provided for token entropy, returning 0")
                return 0.0
            return self.estimators["token_entropy"].compute(token_logprobs)
        
        elif method == "dispersion":
            return self.estimators["dispersion"].compute(generations)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_all(
        self,
        generations: List[str],
        token_logprobs: List[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute all initialized uncertainty measures.
        
        Args:
            generations: List of generated texts
            token_logprobs: Token log probabilities
        
        Returns:
            Dictionary mapping method names to uncertainty values
        """
        results = {}
        
        for method in self.methods:
            try:
                uncertainty = self.compute(generations, token_logprobs, method)
                results[method] = uncertainty
            except Exception as e:
                logger.warning(f"Failed to compute {method}: {e}")
                results[method] = 0.0
        
        return results
    
    def compute_batch(
        self,
        generations_list: List[List[str]],
        token_logprobs_list: List[List[List[float]]] = None,
        method: str = None,
    ) -> List[float]:
        """Compute uncertainty for a batch.
        
        Args:
            generations_list: List of generation lists
            token_logprobs_list: List of token logprobs
            method: Specific method to use
        
        Returns:
            List of uncertainty values
        """
        if method is None:
            method = self.methods[0]
        
        results = []
        for i, generations in enumerate(generations_list):
            token_logprobs = token_logprobs_list[i] if token_logprobs_list else None
            uncertainty = self.compute(generations, token_logprobs, method)
            results.append(uncertainty)
        
        return results
    
    def compute_batch_all(
        self,
        generations_list: List[List[str]],
        token_logprobs_list: List[List[List[float]]] = None,
    ) -> List[Dict[str, float]]:
        """Compute all uncertainty measures for a batch.
        
        Args:
            generations_list: List of generation lists
            token_logprobs_list: List of token logprobs
        
        Returns:
            List of dictionaries with all uncertainty values
        """
        logger.info(f"Computing all uncertainty measures for {len(generations_list)} samples")
        
        results = []
        for i, generations in enumerate(generations_list):
            token_logprobs = token_logprobs_list[i] if token_logprobs_list else None
            uncertainties = self.compute_all(generations, token_logprobs)
            results.append(uncertainties)
        
        return results


def create_uncertainty_estimator(
    methods: List[str] = None,
    **kwargs,
) -> UncertaintyEstimator:
    """Factory function to create uncertainty estimator.
    
    Args:
        methods: List of methods to initialize (default: all)
        **kwargs: Additional configuration for specific estimators
    
    Returns:
        UncertaintyEstimator instance
    """
    if methods is None:
        methods = ["semantic_entropy", "token_entropy", "dispersion"]
    
    return UncertaintyEstimator(methods=methods, **kwargs)


if __name__ == "__main__":
    # Example usage
    estimator = create_uncertainty_estimator()
    
    # Test generations
    generations = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The answer is Paris.",
        "I don't know the answer.",
    ]
    
    token_logprobs = [
        [-0.1, -0.05, -0.02, -0.01],
        [-0.15, -0.08, -0.03, -0.02],
        [-0.2, -0.1, -0.05],
        [-0.8, -0.6, -0.4],
    ]
    
    # Compute all uncertainties
    uncertainties = estimator.compute_all(generations, token_logprobs)
    
    print("Computed uncertainties:")
    for method, value in uncertainties.items():
        print(f"  {method}: {value:.3f}")
    
    # Compute specific uncertainty
    semantic_ent = estimator.compute(generations, method="semantic_entropy")
    print(f"\nSemantic entropy: {semantic_ent:.3f}")

