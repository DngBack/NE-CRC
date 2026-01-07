"""Shift-Aware UniCR (S-UniCR): Full system with SConU filter + NE-CRC."""

from typing import Optional, Callable
import numpy as np
from loguru import logger

from ..data import EvidenceFeatures, PredictionResult
from ..calibration import LogisticCalibrationHead, MLPCalibrationHead
from ..conformal import NonExchangeableCRC
from ..filtering import SConUFilter


class ShiftAwareUniCR:
    """S-UniCR: Full system with SConU filtering + NE-CRC."""
    
    def __init__(
        self,
        calibration_head_type: str = "logistic",
        alpha: float = 0.05,
        delta: float = 0.05,
        weight_fn: Optional[Callable] = None,
        **head_kwargs,
    ):
        """Initialize S-UniCR.
        
        Args:
            calibration_head_type: "logistic" or "mlp"
            alpha: Risk level for NE-CRC
            delta: Outlier threshold for SConU filter
            weight_fn: Weight function for NE-CRC
            **head_kwargs: Additional arguments for calibration head
        """
        self.calibration_head_type = calibration_head_type
        self.alpha = alpha
        self.delta = delta
        
        # Create calibration head
        if calibration_head_type == "logistic":
            from ..calibration import create_logistic_head
            self.calibration_head = create_logistic_head(**head_kwargs)
        elif calibration_head_type == "mlp":
            from ..calibration import create_mlp_head
            self.calibration_head = create_mlp_head(**head_kwargs)
        else:
            raise ValueError(f"Unknown calibration head type: {calibration_head_type}")
        
        # Create NE-CRC
        self.ne_crc = NonExchangeableCRC(alpha=alpha, weight_fn=weight_fn)
        
        # Create SConU filter
        self.filter = SConUFilter(delta=delta)
        
        self.fitted = False
    
    def fit(
        self,
        train_features: list[EvidenceFeatures],
        train_labels: list[float],
        cal_features: list[EvidenceFeatures],
        cal_labels: list[float],
        cal_uncertainties: np.ndarray,
        cal_weight_features: Optional[np.ndarray] = None,
    ):
        """Fit calibration head, NE-CRC, and outlier filter.
        
        Args:
            train_features: Training evidence features
            train_labels: Training correctness labels
            cal_features: Calibration evidence features
            cal_labels: Calibration correctness labels
            cal_uncertainties: Calibration uncertainties for filter
            cal_weight_features: Optional features for weight computation
        """
        logger.info("Fitting S-UniCR (Shift-Aware UniCR)")
        
        # Fit calibration head
        logger.info("Training calibration head")
        self.calibration_head.fit(train_features, train_labels)
        
        # Get calibration confidences
        logger.info("Computing calibration confidences")
        cal_confidences = self.calibration_head.predict_proba(cal_features)
        
        # Calibrate NE-CRC
        logger.info("Calibrating NE-CRC")
        self.ne_crc.calibrate(cal_confidences, np.array(cal_labels), cal_weight_features)
        
        # Calibrate filter
        logger.info("Calibrating SConU filter")
        self.filter.calibrate(cal_uncertainties)
        
        self.fitted = True
        logger.info("S-UniCR fitted successfully")
    
    def predict(
        self,
        test_features: list[EvidenceFeatures],
        test_uncertainties: np.ndarray,
        test_weight_features: Optional[np.ndarray] = None,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Make predictions with filtering and NE-CRC.
        
        Args:
            test_features: Test evidence features
            test_uncertainties: Test uncertainties for filtering
            test_weight_features: Optional features for weight computation
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        if not self.fitted:
            raise RuntimeError("Must fit S-UniCR before prediction")
        
        n = len(test_features)
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        # Get confidences from calibration head
        confidences = self.calibration_head.predict_proba(test_features)
        
        # Filter outliers
        abstain_mask, p_values = self.filter.filter_batch(test_uncertainties)
        
        # Get test-specific thresholds from NE-CRC
        thresholds = self.ne_crc.predict(
            confidences,
            test_weight_features,
            return_decisions=False
        )
        
        # Create results
        results = []
        
        for i, (confidence, threshold, is_outlier, p_value, sample_id, query, uncertainty) in enumerate(
            zip(confidences, thresholds, abstain_mask, p_values, sample_ids, queries, test_uncertainties)
        ):
            if is_outlier:
                # Filtered out as outlier
                decision = "abstain"
                reason = f"Outlier: p-value {p_value:.3f} < delta {self.delta} (SConU filter)"
            else:
                # Apply NE-CRC threshold
                if confidence >= threshold:
                    decision = "answer"
                    reason = f"Confidence {confidence:.3f} ≥ adaptive threshold {threshold:.3f} (NE-CRC), p={p_value:.3f}"
                else:
                    decision = "abstain"
                    reason = f"Confidence {confidence:.3f} < adaptive threshold {threshold:.3f} (NE-CRC), p={p_value:.3f}"
            
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                decision=decision,
                decision_reason=reason,
                threshold=float(threshold),
                p_value=float(p_value),
            )
            
            results.append(result)
        
        return results


def create_s_unicr(
    calibration_head_type: str = "logistic",
    alpha: float = 0.05,
    delta: float = 0.05,
    weight_fn: Optional[Callable] = None,
    **kwargs,
) -> ShiftAwareUniCR:
    """Factory function to create S-UniCR.
    
    Args:
        calibration_head_type: Type of calibration head
        alpha: Risk level
        delta: Outlier threshold
        weight_fn: Weight function for NE-CRC
        **kwargs: Additional arguments
    
    Returns:
        ShiftAwareUniCR instance
    """
    return ShiftAwareUniCR(
        calibration_head_type=calibration_head_type,
        alpha=alpha,
        delta=delta,
        weight_fn=weight_fn,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    from ..data import EvidenceFeatures
    import numpy as np
    
    np.random.seed(42)
    
    # Generate synthetic features
    def generate_features(n):
        features = []
        labels = []
        for i in range(n):
            consistency = np.random.rand()
            f = EvidenceFeatures(
                sample_id=f"sample_{i}",
                consistency_score=consistency,
                semantic_entropy=np.random.rand() * 2,
                token_entropy=np.random.rand(),
                max_token_prob=0.5 + np.random.rand() * 0.5,
                mean_token_prob=0.5 + np.random.rand() * 0.3,
                dispersion=np.random.rand() * 3,
                verbal_confidence=np.random.rand(),
                generation_length=10 + int(np.random.rand() * 20),
                num_unique_answers=1 + int(np.random.rand() * 5),
            )
            label = 1.0 if consistency > 0.5 else 0.0
            features.append(f)
            labels.append(label)
        return features, labels
    
    # Generate data
    train_features, train_labels = generate_features(100)
    cal_features, cal_labels = generate_features(50)
    test_features, test_labels = generate_features(30)
    
    # Generate uncertainties and weight features
    cal_uncertainties = np.random.rand(50) * 2
    test_uncertainties = np.random.rand(30) * 2
    
    cal_weight_features = np.random.randn(50, 10)
    test_weight_features = np.random.randn(30, 10)
    
    # Define weight function
    from ..conformal.weights import create_similarity_weights
    weight_fn = create_similarity_weights()
    
    # Create and fit S-UniCR
    s_unicr = create_s_unicr(
        calibration_head_type="logistic",
        alpha=0.05,
        delta=0.05,
        weight_fn=weight_fn
    )
    
    s_unicr.fit(
        train_features, train_labels,
        cal_features, cal_labels,
        cal_uncertainties,
        cal_weight_features
    )
    
    # Predict
    results = s_unicr.predict(
        test_features,
        test_uncertainties,
        test_weight_features
    )
    
    print(f"\nS-UniCR Predictions:")
    answered = sum(1 for r in results if r.decision == "answer")
    outliers = sum(1 for r in results if "Outlier" in r.decision_reason)
    print(f"  Answered: {answered}/{len(results)}")
    print(f"  Outliers filtered: {outliers}/{len(results)}")
    print(f"  Coverage: {answered / len(results):.2%}")
    
    # Show first few results
    for result in results[:5]:
        print(f"  {result.sample_id}: conf={result.confidence:.3f}, thresh={result.threshold:.3f}, p={result.p_value:.3f} -> {result.decision}")

