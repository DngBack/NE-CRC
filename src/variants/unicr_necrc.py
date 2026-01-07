"""UniCR + NE-CRC variant (no filter)."""

from typing import Optional, Callable
import numpy as np
from loguru import logger

from ..data import EvidenceFeatures, PredictionResult
from ..calibration import LogisticCalibrationHead, MLPCalibrationHead
from ..conformal import NonExchangeableCRC


class UniCRWithNECRC:
    """UniCR + NE-CRC (without filtering)."""
    
    def __init__(
        self,
        calibration_head_type: str = "logistic",
        alpha: float = 0.05,
        weight_fn: Optional[Callable] = None,
        **head_kwargs,
    ):
        """Initialize UniCR with NE-CRC.
        
        Args:
            calibration_head_type: "logistic" or "mlp"
            alpha: Risk level for CRC
            weight_fn: Weight function for NE-CRC
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
        
        # Create NE-CRC
        self.ne_crc = NonExchangeableCRC(alpha=alpha, weight_fn=weight_fn)
        
        self.fitted = False
    
    def fit(
        self,
        train_features: list[EvidenceFeatures],
        train_labels: list[float],
        cal_features: list[EvidenceFeatures],
        cal_labels: list[float],
        cal_weight_features: Optional[np.ndarray] = None,
    ):
        """Fit calibration head and NE-CRC.
        
        Args:
            train_features: Training evidence features
            train_labels: Training correctness labels
            cal_features: Calibration evidence features
            cal_labels: Calibration correctness labels
            cal_weight_features: Optional features for weight computation
        """
        logger.info("Fitting UniCR with NE-CRC")
        
        # Fit calibration head
        logger.info("Training calibration head")
        self.calibration_head.fit(train_features, train_labels)
        
        # Get calibration confidences
        logger.info("Computing calibration confidences")
        cal_confidences = self.calibration_head.predict_proba(cal_features)
        
        # Calibrate NE-CRC
        logger.info("Calibrating NE-CRC")
        self.ne_crc.calibrate(cal_confidences, np.array(cal_labels), cal_weight_features)
        
        self.fitted = True
        logger.info("UniCR + NE-CRC fitted")
    
    def predict(
        self,
        test_features: list[EvidenceFeatures],
        test_weight_features: Optional[np.ndarray] = None,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Make predictions with NE-CRC.
        
        Args:
            test_features: Test evidence features
            test_weight_features: Optional features for weight computation
            sample_ids: Optional sample IDs
            queries: Optional query strings
        
        Returns:
            List of PredictionResult objects
        """
        if not self.fitted:
            raise RuntimeError("Must fit before prediction")
        
        n = len(test_features)
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n)]
        
        if queries is None:
            queries = [""] * n
        
        # Get confidences from calibration head
        confidences = self.calibration_head.predict_proba(test_features)
        
        # Get test-specific thresholds from NE-CRC
        thresholds = self.ne_crc.predict(
            confidences,
            test_weight_features,
            return_decisions=False
        )
        
        # Create results
        results = []
        
        for i, (confidence, threshold, sample_id, query) in enumerate(
            zip(confidences, thresholds, sample_ids, queries)
        ):
            decision = "answer" if confidence >= threshold else "abstain"
            reason = f"Confidence {confidence:.3f} {'≥' if confidence >= threshold else '<'} adaptive threshold {threshold:.3f} (NE-CRC)"
            
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=float(confidence),
                decision=decision,
                decision_reason=reason,
                threshold=float(threshold),
            )
            
            results.append(result)
        
        return results


def create_unicr_with_necrc(
    calibration_head_type: str = "logistic",
    alpha: float = 0.05,
    weight_fn: Optional[Callable] = None,
    **kwargs,
) -> UniCRWithNECRC:
    """Factory function to create UniCR with NE-CRC.
    
    Args:
        calibration_head_type: Type of calibration head
        alpha: Risk level
        weight_fn: Weight function for NE-CRC
        **kwargs: Additional arguments
    
    Returns:
        UniCRWithNECRC instance
    """
    return UniCRWithNECRC(
        calibration_head_type=calibration_head_type,
        alpha=alpha,
        weight_fn=weight_fn,
        **kwargs
    )

