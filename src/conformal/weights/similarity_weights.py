"""Similarity-based weights for NE-CRC."""

import numpy as np
from typing import Optional
from loguru import logger


class SimilarityWeights:
    """Compute similarity-based weights using embeddings."""
    
    def __init__(
        self,
        metric: str = "cosine",
        temperature: float = 1.0,
        normalize: bool = True,
    ):
        """Initialize similarity weights.
        
        Args:
            metric: Similarity metric ("cosine", "euclidean", "dot_product")
            temperature: Temperature for exponential scaling
            normalize: Whether to normalize weights to sum to 1
        """
        self.metric = metric
        self.temperature = temperature
        self.normalize = normalize
        
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Choose from {valid_metrics}")
    
    def compute(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity weights.
        
        Args:
            cal_features: Calibration features [n_cal, feature_dim]
            test_feature: Test feature [feature_dim]
        
        Returns:
            Weights [n_cal]
        """
        if cal_features is None or test_feature is None:
            # Fallback to uniform weights
            n_cal = len(cal_features) if cal_features is not None else 1
            return np.ones(n_cal)
        
        # Compute similarities
        if self.metric == "cosine":
            similarities = self._cosine_similarity(cal_features, test_feature)
        elif self.metric == "euclidean":
            similarities = self._euclidean_similarity(cal_features, test_feature)
        elif self.metric == "dot_product":
            similarities = self._dot_product_similarity(cal_features, test_feature)
        
        # Convert similarities to weights with temperature scaling
        weights = np.exp(similarities / self.temperature)
        
        # Normalize
        if self.normalize:
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights
    
    def _cosine_similarity(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity.
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
        
        Returns:
            Similarities [n_cal]
        """
        # Normalize vectors
        cal_norms = np.linalg.norm(cal_features, axis=1, keepdims=True) + 1e-8
        test_norm = np.linalg.norm(test_feature) + 1e-8
        
        cal_features_normed = cal_features / cal_norms
        test_feature_normed = test_feature / test_norm
        
        # Dot product of normalized vectors
        similarities = np.dot(cal_features_normed, test_feature_normed)
        
        return similarities
    
    def _euclidean_similarity(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity from euclidean distance.
        
        Distance is converted to similarity: sim = -dist
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
        
        Returns:
            Similarities [n_cal]
        """
        # Compute euclidean distances
        distances = np.linalg.norm(cal_features - test_feature, axis=1)
        
        # Convert to similarity (negative distance)
        similarities = -distances
        
        return similarities
    
    def _dot_product_similarity(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute dot product similarity.
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
        
        Returns:
            Similarities [n_cal]
        """
        return np.dot(cal_features, test_feature)
    
    def __call__(self, cal_features, test_feature):
        """Make object callable."""
        return self.compute(cal_features, test_feature)


def create_similarity_weights(
    metric: str = "cosine",
    temperature: float = 1.0,
) -> SimilarityWeights:
    """Factory function to create similarity weights.
    
    Args:
        metric: Similarity metric
        temperature: Temperature for scaling
    
    Returns:
        SimilarityWeights instance
    """
    return SimilarityWeights(metric=metric, temperature=temperature)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate random features
    n_cal = 50
    feature_dim = 10
    cal_features = np.random.randn(n_cal, feature_dim)
    test_feature = np.random.randn(feature_dim)
    
    # Test different similarity metrics
    for metric in ["cosine", "euclidean", "dot_product"]:
        weights_fn = create_similarity_weights(metric=metric, temperature=1.0)
        weights = weights_fn(cal_features, test_feature)
        
        print(f"\n{metric} similarity:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights sum: {np.sum(weights):.3f}")
        print(f"  Max weight: {np.max(weights):.4f}")
        print(f"  Min weight: {np.min(weights):.4f}")
        print(f"  Top 5 weights: {weights[np.argsort(weights)[-5:]]}")

