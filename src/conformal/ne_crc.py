"""Non-Exchangeable Conformal Risk Control (NE-CRC) for shift scenarios."""

from typing import List, Tuple, Optional, Callable
import numpy as np
from loguru import logger


class NonExchangeableCRC:
    """NE-CRC with test-time weighted threshold selection."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        weight_fn: Optional[Callable] = None,
    ):
        """Initialize NE-CRC.
        
        Args:
            alpha: Risk level (target error rate)
            weight_fn: Function to compute weights w_i(x) for each test sample
                      Should take (cal_features, test_feature) and return weights
        """
        self.alpha = alpha
        self.weight_fn = weight_fn
        
        # Store calibration data for test-time weighting
        self.cal_confidences = None
        self.cal_correctness = None
        self.cal_features = None
    
    def calibrate(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        features: Optional[np.ndarray] = None,
    ):
        """Store calibration data for test-time threshold computation.
        
        Args:
            confidences: Calibrated confidence scores c(x)
            correctness: Binary correctness labels
            features: Features for weight computation (e.g., embeddings)
        """
        if len(confidences) != len(correctness):
            raise ValueError("Number of confidences must match number of labels")
        
        logger.info(f"Storing calibration data: {len(confidences)} samples, alpha={self.alpha}")
        
        self.cal_confidences = np.array(confidences)
        self.cal_correctness = np.array(correctness)
        self.cal_features = np.array(features) if features is not None else None
        
        # Precompute nonconformity scores for errors
        self.error_mask = (self.cal_correctness == 0)
        
        if not np.any(self.error_mask):
            logger.warning("No errors in calibration set")
    
    def compute_threshold(
        self,
        test_feature: np.ndarray,
    ) -> float:
        """Compute test-specific threshold using NE-CRC.
        
        Args:
            test_feature: Feature vector for test sample
        
        Returns:
            Threshold τ(x) for this test sample
        """
        if self.cal_confidences is None:
            raise RuntimeError("Must calibrate before computing threshold")
        
        if self.weight_fn is None:
            # Fallback to uniform weights (standard CRC)
            weights = np.ones(len(self.cal_confidences))
        else:
            # Compute relevance weights for this test sample
            weights = self.weight_fn(self.cal_features, test_feature)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        # Focus on error samples
        if not np.any(self.error_mask):
            return 0.0
        
        error_confidences = self.cal_confidences[self.error_mask]
        error_weights = weights[self.error_mask]
        
        # Renormalize error weights
        error_weights = error_weights / np.sum(error_weights) if np.sum(error_weights) > 0 else error_weights
        
        # Compute nonconformity scores
        nonconformity_scores = 1.0 - error_confidences
        
        # Weighted quantile: find q such that sum of weights with score ≤ q is ≥ (1-α)
        q = self._weighted_quantile(nonconformity_scores, error_weights, 1.0 - self.alpha)
        
        # Threshold: τ(x) = 1 - q
        threshold = 1.0 - q
        
        return threshold
    
    def predict(
        self,
        confidences: np.ndarray,
        test_features: Optional[np.ndarray] = None,
        return_decisions: bool = True,
    ) -> np.ndarray:
        """Make predictions using test-specific thresholds.
        
        Args:
            confidences: Confidence scores for test samples
            test_features: Features for test samples (required if weight_fn uses features)
            return_decisions: If True, return binary decisions; else return thresholds
        
        Returns:
            Binary decisions (1=answer, 0=abstain) or threshold values
        """
        if self.cal_confidences is None:
            raise RuntimeError("Must calibrate before prediction")
        
        confidences = np.array(confidences)
        n_test = len(confidences)
        
        # Compute threshold for each test sample
        thresholds = np.zeros(n_test)
        
        for i in range(n_test):
            if test_features is not None:
                test_feature = test_features[i]
            else:
                test_feature = None
            
            thresholds[i] = self.compute_threshold(test_feature)
        
        if return_decisions:
            # Answer if confidence >= threshold
            decisions = (confidences >= thresholds).astype(int)
            return decisions
        else:
            return thresholds
    
    def compute_selective_risk(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        test_features: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Compute selective risk and coverage.
        
        Args:
            confidences: Confidence scores
            correctness: Binary correctness labels
            test_features: Features for test samples
        
        Returns:
            Tuple of (selective_risk, coverage)
        """
        decisions = self.predict(confidences, test_features, return_decisions=True)
        answered_mask = (decisions == 1)
        
        if not np.any(answered_mask):
            return 0.0, 0.0
        
        coverage = np.mean(answered_mask)
        answered_correctness = correctness[answered_mask]
        selective_risk = 1.0 - np.mean(answered_correctness)
        
        return selective_risk, coverage
    
    def _weighted_quantile(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        """Compute weighted quantile.
        
        Args:
            values: Array of values
            weights: Array of weights (must sum to 1)
            quantile: Quantile level in [0, 1]
        
        Returns:
            Weighted quantile value
        """
        if len(values) == 0:
            return 0.0
        
        # Sort by values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        
        # Find first index where cumsum >= quantile
        idx = np.searchsorted(cumsum_weights, quantile, side='left')
        
        if idx >= len(sorted_values):
            idx = len(sorted_values) - 1
        
        return sorted_values[idx]


def create_ne_crc(
    alpha: float = 0.05,
    weight_fn: Optional[Callable] = None,
) -> NonExchangeableCRC:
    """Factory function to create NE-CRC.
    
    Args:
        alpha: Risk level
        weight_fn: Weight function
    
    Returns:
        NonExchangeableCRC instance
    """
    return NonExchangeableCRC(alpha=alpha, weight_fn=weight_fn)


if __name__ == "__main__":
    # Example usage with simple similarity-based weighting
    np.random.seed(42)
    
    # Generate synthetic calibration data with features
    n_cal = 100
    cal_features = np.random.randn(n_cal, 10)  # 10-dim features
    confidences_cal = np.random.rand(n_cal)
    correctness_cal = (confidences_cal > 0.5).astype(int)
    correctness_cal[np.random.rand(n_cal) < 0.2] = 1 - correctness_cal[np.random.rand(n_cal) < 0.2]
    
    # Define simple weight function (cosine similarity)
    def similarity_weights(cal_features, test_feature):
        """Compute cosine similarity weights."""
        if cal_features is None or test_feature is None:
            return np.ones(len(cal_features))
        
        # Cosine similarity
        similarities = np.dot(cal_features, test_feature) / (
            np.linalg.norm(cal_features, axis=1) * np.linalg.norm(test_feature) + 1e-8
        )
        
        # Convert to weights (higher similarity = higher weight)
        # Use exponential to amplify differences
        weights = np.exp(similarities)
        return weights
    
    # Calibrate NE-CRC
    ne_crc = create_ne_crc(alpha=0.05, weight_fn=similarity_weights)
    ne_crc.calibrate(confidences_cal, correctness_cal, cal_features)
    
    # Test data
    n_test = 10
    test_features = np.random.randn(n_test, 10)
    confidences_test = np.random.rand(n_test)
    correctness_test = (confidences_test > 0.5).astype(int)
    
    # Predictions with test-specific thresholds
    thresholds = ne_crc.predict(confidences_test, test_features, return_decisions=False)
    decisions = ne_crc.predict(confidences_test, test_features, return_decisions=True)
    
    print("Test-specific thresholds:")
    for i in range(n_test):
        print(f"  Sample {i}: threshold={thresholds[i]:.4f}, confidence={confidences_test[i]:.4f}, decision={decisions[i]}")
    
    # Compute selective risk
    selective_risk, coverage = ne_crc.compute_selective_risk(
        confidences_test, correctness_test, test_features
    )
    print(f"\nTest set:")
    print(f"  Coverage: {coverage:.3f}")
    print(f"  Selective risk: {selective_risk:.3f}")

