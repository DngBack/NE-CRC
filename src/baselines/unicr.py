"""UniCR baseline with standard CRC."""

from typing import Optional, Any
import numpy as np
from loguru import logger

from ..data import EvidenceFeatures, PredictionResult
from ..calibration import LogisticCalibrationHead, MLPCalibrationHead
from ..conformal import StandardCRC


class UniCRBaseline:
    """UniCR baseline: Calibration head + Standard CRC."""
    
    def __init__(
        self,
        calibration_head_type: str = "logistic",
        alpha: float = 0.05,
        **head_kwargs,
    ):
        """Initialize UniCR baseline.
        
        Args:
            calibration_head_type: "logistic" or "mlp"
            alpha: Risk level for CRC
            **head_kwargs: Additional arguments for calibration head
        """
        self.calibration_head_type = calibration_head_type
        self.alpha = alpha
        
        # Create calibration head
        if calibration_head_type == "logistic":
            from ..calibration import create_logistic_head
            self.calibration_head = create_logistic_head(**head_kwargs)
        elif calibration_head_type == "mlp":
            from ..calibration import create_mlp_head
            self.calibration_head = create_mlp_head(**head_kwargs)
        else:
            raise ValueError(f"Unknown calibration head type: {calibration_head_type}")
        
        # Create CRC
        self.crc = StandardCRC(alpha=alpha)
        
        self.fitted = False
    
    def fit(
        self,
        train_features: list[EvidenceFeatures],
        train_labels: list[float],
        cal_features: list[EvidenceFeatures],
        cal_labels: list[float],
    ):
        """Fit calibration head and calibrate CRC threshold.
        
        Args:
            train_features: Training evidence features
            train_labels: Training correctness labels
            cal_features: Calibration evidence features
            cal_labels: Calibration correctness labels
        """
        logger.info("Fitting UniCR baseline")
        
        # Fit calibration head
        logger.info("Training calibration head")
        self.calibration_head.fit(train_features, train_labels)
        
        # Get calibration confidences
        logger.info("Computing calibration confidences")
        cal_confidences = self.calibration_head.predict_proba(cal_features)
        
        # Calibrate CRC threshold
        logger.info("Calibrating CRC threshold")
        self.crc.calibrate(cal_confidences, np.array(cal_labels))
        
        self.fitted = True
        logger.info(f"UniCR fitted. Threshold: {self.crc.get_threshold():.4f}")
    
    def predict(
        self,
        test_features: list[EvidenceFeatures],
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Make predictions on test set.
        
        Args:
            test_features: Test evidence features
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        if not self.fitted:
            raise RuntimeError("Must fit UniCR before prediction")
        
        n = len(test_features)
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        # Get confidences from calibration head
        confidences = self.calibration_head.predict_proba(test_features)
        
        # Get CRC decisions
        decisions = self.crc.predict(confidences, return_decisions=True)
        threshold = self.crc.get_threshold()
        
        # Create results
        results = []
        
        for i, (confidence, decision, sample_id, query) in enumerate(
            zip(confidences, decisions, sample_ids, queries)
        ):
            decision_str = "answer" if decision == 1 else "abstain"
            reason = f"Confidence {confidence:.3f} {'≥' if decision == 1 else '<'} threshold {threshold:.3f} (CRC)"
            
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=float(confidence),
                decision=decision_str,
                decision_reason=reason,
                threshold=threshold,
            )
            
            results.append(result)
        
        return results


def create_unicr_baseline(
    calibration_head_type: str = "logistic",
    alpha: float = 0.05,
    **kwargs,
) -> UniCRBaseline:
    """Factory function to create UniCR baseline.
    
    Args:
        calibration_head_type: Type of calibration head
        alpha: Risk level
        **kwargs: Additional arguments
    
    Returns:
        UniCRBaseline instance
    """
    return UniCRBaseline(calibration_head_type=calibration_head_type, alpha=alpha, **kwargs)


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
    
    # Create and fit UniCR
    unicr = create_unicr_baseline(calibration_head_type="logistic", alpha=0.05)
    unicr.fit(train_features, train_labels, cal_features, cal_labels)
    
    # Predict
    results = unicr.predict(test_features)
    
    print(f"\nUniCR Predictions:")
    answered = sum(1 for r in results if r.decision == "answer")
    print(f"  Answered: {answered}/{len(results)}")
    print(f"  Coverage: {answered / len(results):.2%}")
    
    # Show first few results
    for result in results[:5]:
        print(f"  {result.sample_id}: conf={result.confidence:.3f} -> {result.decision}")

