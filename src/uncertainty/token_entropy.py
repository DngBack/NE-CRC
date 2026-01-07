"""Token-level entropy computation for uncertainty quantification."""

from typing import List, Optional
import numpy as np
from loguru import logger


class TokenEntropyEstimator:
    """Compute token-level entropy for uncertainty quantification."""
    
    def __init__(self, aggregation: str = "mean"):
        """Initialize token entropy estimator.
        
        Args:
            aggregation: How to aggregate token entropies ("mean", "max", "min", "first")
        """
        self.aggregation = aggregation
        
        valid_aggs = ["mean", "max", "min", "first", "sum"]
        if aggregation not in valid_aggs:
            raise ValueError(f"Invalid aggregation: {aggregation}. Choose from {valid_aggs}")
    
    def compute(
        self,
        token_logprobs: List[List[float]],
    ) -> float:
        """Compute token entropy from token log probabilities.
        
        Args:
            token_logprobs: List of token log probabilities for K samples
                           Each element is a list of logprobs for one generation
        
        Returns:
            Aggregated token entropy
        """
        if not token_logprobs:
            logger.warning("No token logprobs provided, returning 0")
            return 0.0
        
        # Compute entropy for each sample
        sample_entropies = []
        
        for sample_logprobs in token_logprobs:
            if not sample_logprobs:
                continue
            
            # Convert logprobs to probabilities
            probs = np.exp(sample_logprobs)
            
            # Clip to avoid numerical issues
            probs = np.clip(probs, 1e-10, 1.0)
            
            # Compute per-token entropy: -p * log(p)
            # Note: This is entropy of each token's distribution (1 token = 1 distribution)
            # For multiple tokens, we aggregate
            token_entropies = -probs * np.log(probs)
            
            # Aggregate across tokens in this sample
            if self.aggregation == "mean":
                sample_entropy = np.mean(token_entropies)
            elif self.aggregation == "max":
                sample_entropy = np.max(token_entropies)
            elif self.aggregation == "min":
                sample_entropy = np.min(token_entropies)
            elif self.aggregation == "first":
                sample_entropy = token_entropies[0] if len(token_entropies) > 0 else 0.0
            elif self.aggregation == "sum":
                sample_entropy = np.sum(token_entropies)
            
            sample_entropies.append(sample_entropy)
        
        # Aggregate across samples
        if not sample_entropies:
            return 0.0
        
        return float(np.mean(sample_entropies))
    
    def compute_batch(
        self,
        token_logprobs_list: List[List[List[float]]],
    ) -> List[float]:
        """Compute token entropy for a batch.
        
        Args:
            token_logprobs_list: List of token_logprobs (one per sample)
        
        Returns:
            List of token entropy values
        """
        logger.info(f"Computing token entropy for {len(token_logprobs_list)} samples")
        return [self.compute(logprobs) for logprobs in token_logprobs_list]
    
    def compute_distribution_entropy(
        self,
        token_logprobs: List[List[float]],
    ) -> float:
        """Compute entropy of the distribution over next tokens.
        
        This averages the predictive distribution across samples and computes
        entropy of that averaged distribution.
        
        Args:
            token_logprobs: List of token log probabilities for K samples
        
        Returns:
            Entropy of averaged predictive distribution
        """
        if not token_logprobs:
            return 0.0
        
        # Find max length
        max_len = max(len(logprobs) for logprobs in token_logprobs if logprobs)
        
        if max_len == 0:
            return 0.0
        
        # Compute entropy at each position
        position_entropies = []
        
        for pos in range(max_len):
            # Get probabilities at this position from all samples
            pos_probs = []
            for sample_logprobs in token_logprobs:
                if pos < len(sample_logprobs):
                    prob = np.exp(sample_logprobs[pos])
                    pos_probs.append(prob)
            
            if pos_probs:
                # Average probability at this position
                avg_prob = np.mean(pos_probs)
                avg_prob = np.clip(avg_prob, 1e-10, 1.0)
                
                # Entropy: -p * log(p)
                entropy = -avg_prob * np.log(avg_prob)
                position_entropies.append(entropy)
        
        return float(np.mean(position_entropies)) if position_entropies else 0.0
    
    def compute_predictive_entropy(
        self,
        token_logprobs: List[List[float]],
    ) -> float:
        """Compute predictive entropy (entropy of the mixture distribution).
        
        This is a different formulation: first average probabilities across samples,
        then compute entropy of that averaged distribution.
        
        Args:
            token_logprobs: List of token log probabilities for K samples
        
        Returns:
            Predictive entropy
        """
        return self.compute_distribution_entropy(token_logprobs)


def create_token_entropy_estimator(
    aggregation: str = "mean",
) -> TokenEntropyEstimator:
    """Factory function to create token entropy estimator.
    
    Args:
        aggregation: Aggregation method
    
    Returns:
        TokenEntropyEstimator instance
    """
    return TokenEntropyEstimator(aggregation=aggregation)


if __name__ == "__main__":
    # Example usage
    estimator = create_token_entropy_estimator(aggregation="mean")
    
    # Test token logprobs (3 samples, varying lengths)
    token_logprobs = [
        [-0.1, -0.05, -0.02, -0.01],  # High confidence (close to 0)
        [-0.8, -0.6, -0.4, -0.3],     # Lower confidence
        [-0.5, -0.4, -0.3],            # Medium confidence
    ]
    
    entropy = estimator.compute(token_logprobs)
    print(f"Token entropy (mean): {entropy:.3f}")
    
    # Predictive entropy
    pred_entropy = estimator.compute_predictive_entropy(token_logprobs)
    print(f"Predictive entropy: {pred_entropy:.3f}")
    
    # Test different aggregations
    for agg in ["mean", "max", "min", "first"]:
        est = create_token_entropy_estimator(aggregation=agg)
        ent = est.compute(token_logprobs)
        print(f"Token entropy ({agg}): {ent:.3f}")

