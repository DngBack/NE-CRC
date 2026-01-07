"""Density ratio weights for NE-CRC using domain classification."""

import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger


class DensityRatioWeights:
    """Compute importance weights via density ratio estimation.
    
    Uses a domain classifier to estimate p_test(x) / p_cal(x).
    """
    
    def __init__(
        self,
        clip_min: float = 0.1,
        clip_max: float = 10.0,
        normalize: bool = True,
    ):
        """Initialize density ratio weights.
        
        Args:
            clip_min: Minimum weight value (for stability)
            clip_max: Maximum weight value (for stability)
            normalize: Whether to normalize weights to sum to 1
        """
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize = normalize
        
        self.classifier = None
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(
        self,
        cal_features: np.ndarray,
        test_features: np.ndarray,
    ):
        """Fit domain classifier on calibration and test features.
        
        Args:
            cal_features: Calibration features [n_cal, feature_dim]
            test_features: Test features [n_test, feature_dim]
        """
        logger.info(f"Fitting domain classifier: {len(cal_features)} cal, {len(test_features)} test")
        
        # Create labels: 0 for calibration, 1 for test
        n_cal = len(cal_features)
        n_test = len(test_features)
        
        X = np.vstack([cal_features, test_features])
        y = np.hstack([np.zeros(n_cal), np.ones(n_test)])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit logistic regression as domain classifier
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )
        self.classifier.fit(X_scaled, y)
        
        self.fitted = True
        
        # Log accuracy
        acc = self.classifier.score(X_scaled, y)
        logger.info(f"Domain classifier accuracy: {acc:.3f}")
    
    def compute(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute density ratio weights for a test sample.
        
        Args:
            cal_features: Calibration features [n_cal, feature_dim]
            test_feature: Test feature [feature_dim]
        
        Returns:
            Weights [n_cal]
        """
        if not self.fitted:
            logger.warning("Density ratio classifier not fitted, returning uniform weights")
            return np.ones(len(cal_features))
        
        # Scale features
        cal_features_scaled = self.scaler.transform(cal_features)
        test_feature_scaled = self.scaler.transform(test_feature.reshape(1, -1))
        
        # Get probabilities: p(test domain | x)
        # For calibration samples
        cal_probs = self.classifier.predict_proba(cal_features_scaled)[:, 1]  # p(test|x)
        
        # For test sample
        test_prob = self.classifier.predict_proba(test_feature_scaled)[0, 1]
        
        # Density ratio: p_test(x_cal) / p_cal(x_cal)
        # Using Bayes rule: p_test(x) / p_cal(x) = [p(test|x) / (1-p(test|x))] * [n_cal / n_test]
        # We approximate with just p(test|x) / (1-p(test|x)) for relative weighting
        
        cal_ratios = cal_probs / (1 - cal_probs + 1e-8)
        
        # Weight each calibration sample by how "test-like" it is
        # If test sample is similar to calibration, we want calibration samples similar to test
        # Simple approach: weight by similarity to test distribution
        weights = cal_ratios
        
        # Clip for stability
        weights = np.clip(weights, self.clip_min, self.clip_max)
        
        # Normalize
        if self.normalize:
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights
    
    def __call__(self, cal_features, test_feature):
        """Make object callable."""
        return self.compute(cal_features, test_feature)


def create_density_ratio_weights(
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> DensityRatioWeights:
    """Factory function to create density ratio weights.
    
    Args:
        clip_min: Minimum weight
        clip_max: Maximum weight
    
    Returns:
        DensityRatioWeights instance
    """
    return DensityRatioWeights(clip_min=clip_min, clip_max=clip_max)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate features with shift
    n_cal = 100
    n_test = 50
    feature_dim = 10
    
    # Calibration from one distribution
    cal_features = np.random.randn(n_cal, feature_dim)
    
    # Test from shifted distribution (different mean)
    test_features = np.random.randn(n_test, feature_dim) + 0.5
    
    # Fit density ratio estimator
    weights_fn = create_density_ratio_weights()
    weights_fn.fit(cal_features, test_features)
    
    # Compute weights for a test sample
    test_sample = test_features[0]
    weights = weights_fn(cal_features, test_sample)
    
    print(f"Density ratio weights:")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum: {np.sum(weights):.3f}")
    print(f"  Max weight: {np.max(weights):.4f}")
    print(f"  Min weight: {np.min(weights):.4f}")
    print(f"  Mean weight: {np.mean(weights):.4f}")

