"""UniCR + SConU Filter variant."""

from typing import Optional
import numpy as np
from loguru import logger

from ..data import EvidenceFeatures, PredictionResult
from ..baselines.unicr import UniCRBaseline
from ..filtering import SConUFilter


class UniCRWithFilter(UniCRBaseline):
    """UniCR + SConU filtering (without NE-CRC)."""
    
    def __init__(
        self,
        calibration_head_type: str = "logistic",
        alpha: float = 0.05,
        delta: float = 0.05,
        **head_kwargs,
    ):
        """Initialize UniCR with filtering.
        
        Args:
            calibration_head_type: "logistic" or "mlp"
            alpha: Risk level for CRC
            delta: Outlier threshold for SConU filter
            **head_kwargs: Additional arguments for calibration head
        """
        super().__init__(calibration_head_type, alpha, **head_kwargs)
        
        self.delta = delta
        self.filter = SConUFilter(delta=delta)
    
    def fit(
        self,
        train_features: list[EvidenceFeatures],
        train_labels: list[float],
        cal_features: list[EvidenceFeatures],
        cal_labels: list[float],
        cal_uncertainties: np.ndarray,
    ):
        """Fit calibration head, CRC, and outlier filter.
        
        Args:
            train_features: Training evidence features
            train_labels: Training correctness labels
            cal_features: Calibration evidence features
            cal_labels: Calibration correctness labels
            cal_uncertainties: Calibration uncertainties for filter
        """
        logger.info("Fitting UniCR with SConU filter")
        
        # Fit base UniCR
        super().fit(train_features, train_labels, cal_features, cal_labels)
        
        # Calibrate filter
        logger.info("Calibrating SConU filter")
        self.filter.calibrate(cal_uncertainties)
        
        logger.info("UniCR + Filter fitted")
    
    def predict(
        self,
        test_features: list[EvidenceFeatures],
        test_uncertainties: np.ndarray,
        sample_ids: Optional[list] = None,
        queries: Optional[list] = None,
    ) -> list[PredictionResult]:
        """Make predictions with filtering.
        
        Args:
            test_features: Test evidence features
            test_uncertainties: Test uncertainties for filtering
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
        threshold = self.crc.get_threshold()
        
        # Filter outliers
        abstain_mask, p_values = self.filter.filter_batch(test_uncertainties)
        
        # Create results
        results = []
        
        for i, (confidence, is_outlier, p_value, sample_id, query, uncertainty) in enumerate(
            zip(confidences, abstain_mask, p_values, sample_ids, queries, test_uncertainties)
        ):
            if is_outlier:
                # Filtered out as outlier
                decision = "abstain"
                reason = f"Outlier: p-value {p_value:.3f} < delta {self.delta} (SConU filter)"
            else:
                # Apply CRC threshold
                if confidence >= threshold:
                    decision = "answer"
                    reason = f"Confidence {confidence:.3f} ≥ threshold {threshold:.3f} (CRC), p={p_value:.3f}"
                else:
                    decision = "abstain"
                    reason = f"Confidence {confidence:.3f} < threshold {threshold:.3f} (CRC), p={p_value:.3f}"
            
            result = PredictionResult(
                sample_id=sample_id,
                query=query,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                decision=decision,
                decision_reason=reason,
                threshold=threshold,
                p_value=float(p_value),
            )
            
            results.append(result)
        
        return results


def create_unicr_with_filter(
    calibration_head_type: str = "logistic",
    alpha: float = 0.05,
    delta: float = 0.05,
    **kwargs,
) -> UniCRWithFilter:
    """Factory function to create UniCR with filter.
    
    Args:
        calibration_head_type: Type of calibration head
        alpha: Risk level
        delta: Outlier threshold
        **kwargs: Additional arguments
    
    Returns:
        UniCRWithFilter instance
    """
    return UniCRWithFilter(
        calibration_head_type=calibration_head_type,
        alpha=alpha,
        delta=delta,
        **kwargs
    )

