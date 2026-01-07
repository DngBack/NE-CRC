"""Standard Conformal Risk Control (CRC) for UniCR baseline."""

from typing import List, Tuple, Optional
import numpy as np
from loguru import logger


class StandardCRC:
    """Standard CRC threshold selection under exchangeability assumption."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize standard CRC.
        
        Args:
            alpha: Risk level (target error rate)
        """
        self.alpha = alpha
        self.threshold = None
    
    def calibrate(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> float:
        """Calibrate threshold using CRC on calibration set.
        
        Args:
            confidences: Calibrated confidence scores c(x) from calibration head
            correctness: Binary correctness labels (1=correct, 0=incorrect)
        
        Returns:
            Threshold τ for decision rule
        """
        if len(confidences) != len(correctness):
            raise ValueError("Number of confidences must match number of labels")
        
        logger.info(f"Calibrating CRC threshold with {len(confidences)} samples, alpha={self.alpha}")
        
        # Convert to numpy arrays
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Compute nonconformity scores: φ(x) = 1 - c(x) for error cases
        # CRC uses monotone loss, we focus on errors (correctness=0)
        error_mask = (correctness == 0)
        
        if not np.any(error_mask):
            logger.warning("No errors in calibration set, setting threshold to 0")
            self.threshold = 0.0
            return self.threshold
        
        error_confidences = confidences[error_mask]
        nonconformity_scores = 1.0 - error_confidences
        
        # Compute (1-α) quantile of nonconformity scores
        # This ensures that at most α fraction of calibration errors have φ(x) ≤ threshold
        quantile_level = 1.0 - self.alpha
        q = np.quantile(nonconformity_scores, quantile_level)
        
        # Threshold: τ = 1 - q
        # Answer if c(x) ≥ τ (i.e., 1 - c(x) ≤ q)
        self.threshold = 1.0 - q
        
        logger.info(f"CRC threshold: {self.threshold:.4f}")
        
        return self.threshold
    
    def predict(
        self,
        confidences: np.ndarray,
        return_decisions: bool = True,
    ) -> np.ndarray:
        """Make predictions using calibrated threshold.
        
        Args:
            confidences: Confidence scores for test samples
            return_decisions: If True, return binary decisions; else return thresholds
        
        Returns:
            Binary decisions (1=answer, 0=abstain) or threshold values
        """
        if self.threshold is None:
            raise RuntimeError("Must calibrate threshold before prediction")
        
        confidences = np.array(confidences)
        
        if return_decisions:
            # Answer if confidence >= threshold
            decisions = (confidences >= self.threshold).astype(int)
            return decisions
        else:
            # Return threshold for each sample (constant in standard CRC)
            return np.full(len(confidences), self.threshold)
    
    def compute_selective_risk(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute selective risk and coverage.
        
        Args:
            confidences: Confidence scores
            correctness: Binary correctness labels
        
        Returns:
            Tuple of (selective_risk, coverage)
        """
        if self.threshold is None:
            raise RuntimeError("Must calibrate threshold before computing risk")
        
        decisions = self.predict(confidences)
        answered_mask = (decisions == 1)
        
        if not np.any(answered_mask):
            # No samples answered
            return 0.0, 0.0
        
        # Coverage: fraction answered
        coverage = np.mean(answered_mask)
        
        # Selective risk: error rate on answered samples
        answered_correctness = correctness[answered_mask]
        selective_risk = 1.0 - np.mean(answered_correctness)
        
        return selective_risk, coverage
    
    def get_threshold(self) -> Optional[float]:
        """Get the calibrated threshold.
        
        Returns:
            Threshold value or None if not calibrated
        """
        return self.threshold


def create_standard_crc(alpha: float = 0.05) -> StandardCRC:
    """Factory function to create standard CRC.
    
    Args:
        alpha: Risk level
    
    Returns:
        StandardCRC instance
    """
    return StandardCRC(alpha=alpha)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic calibration data
    n_cal = 100
    confidences_cal = np.random.rand(n_cal)
    # Correctness correlated with confidence
    correctness_cal = (confidences_cal > 0.5).astype(int)
    
    # Add some noise
    flip_mask = np.random.rand(n_cal) < 0.2
    correctness_cal[flip_mask] = 1 - correctness_cal[flip_mask]
    
    # Calibrate CRC
    crc = create_standard_crc(alpha=0.05)
    threshold = crc.calibrate(confidences_cal, correctness_cal)
    
    print(f"Calibrated threshold: {threshold:.4f}")
    
    # Test data
    n_test = 50
    confidences_test = np.random.rand(n_test)
    correctness_test = (confidences_test > 0.5).astype(int)
    correctness_test[np.random.rand(n_test) < 0.2] = 1 - correctness_test[np.random.rand(n_test) < 0.2]
    
    # Predictions
    decisions = crc.predict(confidences_test)
    selective_risk, coverage = crc.compute_selective_risk(confidences_test, correctness_test)
    
    print(f"\nTest set:")
    print(f"  Coverage: {coverage:.3f}")
    print(f"  Selective risk: {selective_risk:.3f}")
    print(f"  Num answered: {np.sum(decisions)}/{len(decisions)}")

