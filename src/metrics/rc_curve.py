"""Risk-Coverage (RC) curve computation and AURC metric."""

from typing import Tuple, List
import numpy as np
from loguru import logger


class RCCurve:
    """Risk-Coverage curve computation."""
    
    @staticmethod
    def compute_rc_curve(
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Risk-Coverage curve.
        
        The RC curve shows the trade-off between selective risk and coverage
        as the confidence threshold is varied.
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels (1=correct, 0=incorrect)
            num_points: Number of points on the curve
        
        Returns:
            Tuple of (coverages, risks) arrays
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_confidences = confidences[sorted_indices]
        sorted_correctness = correctness[sorted_indices]
        
        n = len(confidences)
        
        # Compute risk and coverage at different thresholds
        coverages = []
        risks = []
        
        # Sweep through all possible thresholds
        unique_confidences = np.unique(sorted_confidences)
        thresholds = np.linspace(
            unique_confidences.min(),
            unique_confidences.max(),
            num_points
        )
        
        for threshold in thresholds:
            # Samples with confidence >= threshold
            answered_mask = (confidences >= threshold)
            
            if not np.any(answered_mask):
                coverage = 0.0
                risk = 0.0
            else:
                coverage = np.mean(answered_mask)
                answered_correctness = correctness[answered_mask]
                risk = 1.0 - np.mean(answered_correctness)
            
            coverages.append(coverage)
            risks.append(risk)
        
        # Sort by coverage (for plotting)
        coverages = np.array(coverages)
        risks = np.array(risks)
        sorted_indices = np.argsort(coverages)
        
        return coverages[sorted_indices], risks[sorted_indices]
    
    @staticmethod
    def compute_aurc(
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_points: int = 100,
    ) -> float:
        """Compute Area Under Risk-Coverage curve (AURC).
        
        Lower AURC is better (less risk for same coverage).
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels
            num_points: Number of points for curve computation
        
        Returns:
            AURC value
        """
        coverages, risks = RCCurve.compute_rc_curve(
            confidences, correctness, num_points
        )
        
        # Compute area using trapezoidal rule
        if len(coverages) < 2:
            return 0.0
        
        aurc = np.trapz(risks, coverages)
        
        return float(aurc)
    
    @staticmethod
    def compute_normalized_aurc(
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_points: int = 100,
    ) -> float:
        """Compute normalized AURC.
        
        Normalized by the oracle (perfect confidence) AURC.
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels
            num_points: Number of points
        
        Returns:
            Normalized AURC
        """
        # Compute actual AURC
        aurc = RCCurve.compute_aurc(confidences, correctness, num_points)
        
        # Compute oracle AURC (perfect confidence = correctness)
        oracle_aurc = RCCurve.compute_aurc(correctness, correctness, num_points)
        
        if oracle_aurc == 0:
            return 0.0
        
        normalized_aurc = aurc / oracle_aurc
        
        return float(normalized_aurc)
    
    @staticmethod
    def compute_eaurc(
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_points: int = 100,
    ) -> float:
        """Compute Excess AURC (E-AURC).
        
        E-AURC = AURC - AURC_oracle
        
        Args:
            confidences: Confidence scores
            correctness: Correctness labels
            num_points: Number of points
        
        Returns:
            E-AURC value
        """
        aurc = RCCurve.compute_aurc(confidences, correctness, num_points)
        oracle_aurc = RCCurve.compute_aurc(correctness, correctness, num_points)
        
        eaurc = aurc - oracle_aurc
        
        return float(eaurc)


def compute_rc_metrics(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_points: int = 100,
) -> dict:
    """Compute all RC-based metrics.
    
    Args:
        confidences: Confidence scores
        correctness: Correctness labels
        num_points: Number of points for curve
    
    Returns:
        Dictionary of metrics including curve data
    """
    coverages, risks = RCCurve.compute_rc_curve(confidences, correctness, num_points)
    aurc = RCCurve.compute_aurc(confidences, correctness, num_points)
    normalized_aurc = RCCurve.compute_normalized_aurc(confidences, correctness, num_points)
    eaurc = RCCurve.compute_eaurc(confidences, correctness, num_points)
    
    return {
        'coverages': coverages,
        'risks': risks,
        'aurc': aurc,
        'normalized_aurc': normalized_aurc,
        'eaurc': eaurc,
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 200
    confidences = np.random.rand(n)
    # Make correctness correlated with confidence
    correctness = (confidences + np.random.randn(n) * 0.3 > 0.5).astype(float)
    
    # Compute RC curve
    coverages, risks = RCCurve.compute_rc_curve(confidences, correctness)
    
    print(f"RC Curve computed with {len(coverages)} points")
    print(f"Coverage range: [{coverages.min():.3f}, {coverages.max():.3f}]")
    print(f"Risk range: [{risks.min():.3f}, {risks.max():.3f}]")
    
    # Compute AURC
    aurc = RCCurve.compute_aurc(confidences, correctness)
    print(f"\nAURC: {aurc:.4f}")
    
    # Compute normalized metrics
    normalized_aurc = RCCurve.compute_normalized_aurc(confidences, correctness)
    eaurc = RCCurve.compute_eaurc(confidences, correctness)
    
    print(f"Normalized AURC: {normalized_aurc:.4f}")
    print(f"E-AURC: {eaurc:.4f}")

