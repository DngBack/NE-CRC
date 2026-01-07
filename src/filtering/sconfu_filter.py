"""SConU-style conformal p-value filtering for outlier detection."""

from typing import List, Tuple
import numpy as np
from loguru import logger


class SConUFilter:
    """Conformal p-value based outlier filtering (SConU-style)."""
    
    def __init__(self, delta: float = 0.05):
        """Initialize SConU filter.
        
        Args:
            delta: Outlier threshold (abstain if p-value < delta)
        """
        self.delta = delta
        self.cal_uncertainties = None
    
    def calibrate(
        self,
        uncertainties: np.ndarray,
    ):
        """Store calibration uncertainties for p-value computation.
        
        Args:
            uncertainties: Uncertainty values from calibration set u(x_i)
        """
        logger.info(f"Storing calibration uncertainties: {len(uncertainties)} samples")
        self.cal_uncertainties = np.array(uncertainties)
    
    def compute_pvalue(
        self,
        test_uncertainty: float,
    ) -> float:
        """Compute conformal p-value for a test sample.
        
        P-value = (1 + #{i: u(x_i) >= u(x_test)}) / (m + 1)
        
        Args:
            test_uncertainty: Uncertainty for test sample
        
        Returns:
            P-value in [0, 1]
        """
        if self.cal_uncertainties is None:
            raise RuntimeError("Must calibrate before computing p-values")
        
        m = len(self.cal_uncertainties)
        
        # Count calibration samples with uncertainty >= test uncertainty
        count = np.sum(self.cal_uncertainties >= test_uncertainty)
        
        # Conformal p-value
        p_value = (1 + count) / (m + 1)
        
        return float(p_value)
    
    def should_abstain(
        self,
        test_uncertainty: float,
    ) -> Tuple[bool, float]:
        """Determine if sample should be abstained (outlier).
        
        Args:
            test_uncertainty: Uncertainty for test sample
        
        Returns:
            Tuple of (should_abstain, p_value)
        """
        p_value = self.compute_pvalue(test_uncertainty)
        should_abstain = (p_value < self.delta)
        
        return should_abstain, p_value
    
    def filter_batch(
        self,
        test_uncertainties: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter a batch of test samples.
        
        Args:
            test_uncertainties: Array of test uncertainties
        
        Returns:
            Tuple of (abstain_mask, p_values)
              - abstain_mask: Boolean array (True = outlier/abstain)
              - p_values: Array of p-values
        """
        if self.cal_uncertainties is None:
            raise RuntimeError("Must calibrate before filtering")
        
        logger.info(f"Filtering {len(test_uncertainties)} test samples with delta={self.delta}")
        
        test_uncertainties = np.array(test_uncertainties)
        n_test = len(test_uncertainties)
        
        p_values = np.zeros(n_test)
        abstain_mask = np.zeros(n_test, dtype=bool)
        
        for i in range(n_test):
            abstain, p_val = self.should_abstain(test_uncertainties[i])
            abstain_mask[i] = abstain
            p_values[i] = p_val
        
        n_outliers = np.sum(abstain_mask)
        outlier_rate = n_outliers / n_test
        
        logger.info(f"Detected {n_outliers}/{n_test} outliers ({outlier_rate:.2%})")
        
        return abstain_mask, p_values
    
    def get_outlier_rate(
        self,
        test_uncertainties: np.ndarray,
    ) -> float:
        """Compute outlier rate for test set.
        
        Args:
            test_uncertainties: Array of test uncertainties
        
        Returns:
            Outlier rate (fraction with p-value < delta)
        """
        abstain_mask, _ = self.filter_batch(test_uncertainties)
        return float(np.mean(abstain_mask))
    
    def compute_pvalues_batch(
        self,
        test_uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Compute p-values for a batch without filtering.
        
        Args:
            test_uncertainties: Array of test uncertainties
        
        Returns:
            Array of p-values
        """
        _, p_values = self.filter_batch(test_uncertainties)
        return p_values


class AdaptiveSConUFilter(SConUFilter):
    """Adaptive SConU filter with dynamic delta selection."""
    
    def __init__(
        self,
        delta: float = 0.05,
        max_outlier_rate: float = 0.5,
    ):
        """Initialize adaptive filter.
        
        Args:
            delta: Initial outlier threshold
            max_outlier_rate: Maximum allowed outlier rate (adjust delta if exceeded)
        """
        super().__init__(delta)
        self.initial_delta = delta
        self.max_outlier_rate = max_outlier_rate
    
    def filter_batch_adaptive(
        self,
        test_uncertainties: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Filter with adaptive delta adjustment.
        
        If outlier rate exceeds max_outlier_rate, gradually reduce delta.
        
        Args:
            test_uncertainties: Array of test uncertainties
        
        Returns:
            Tuple of (abstain_mask, p_values, adjusted_delta)
        """
        # Try initial delta
        abstain_mask, p_values = self.filter_batch(test_uncertainties)
        outlier_rate = np.mean(abstain_mask)
        
        # If too many outliers, reduce delta
        if outlier_rate > self.max_outlier_rate:
            logger.info(
                f"Outlier rate {outlier_rate:.2%} exceeds max {self.max_outlier_rate:.2%}, "
                f"adjusting delta"
            )
            
            # Binary search for appropriate delta
            delta_low = 0.0
            delta_high = self.delta
            
            for _ in range(10):  # Max 10 iterations
                delta_mid = (delta_low + delta_high) / 2
                self.delta = delta_mid
                
                abstain_mask, p_values = self.filter_batch(test_uncertainties)
                outlier_rate = np.mean(abstain_mask)
                
                if outlier_rate > self.max_outlier_rate:
                    delta_high = delta_mid
                else:
                    delta_low = delta_mid
                
                if abs(outlier_rate - self.max_outlier_rate) < 0.01:
                    break
            
            logger.info(f"Adjusted delta to {self.delta:.4f} (outlier rate: {outlier_rate:.2%})")
        
        return abstain_mask, p_values, self.delta


def create_sconfu_filter(
    delta: float = 0.05,
    adaptive: bool = False,
) -> SConUFilter:
    """Factory function to create SConU filter.
    
    Args:
        delta: Outlier threshold
        adaptive: Whether to use adaptive delta adjustment
    
    Returns:
        SConUFilter instance
    """
    if adaptive:
        return AdaptiveSConUFilter(delta=delta)
    else:
        return SConUFilter(delta=delta)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate calibration uncertainties (from one distribution)
    n_cal = 100
    cal_uncertainties = np.random.exponential(scale=1.0, size=n_cal)
    
    # Create filter
    sconfu = create_sconfu_filter(delta=0.05)
    sconfu.calibrate(cal_uncertainties)
    
    # Test on in-distribution samples (should have high p-values)
    n_id = 50
    id_uncertainties = np.random.exponential(scale=1.0, size=n_id)
    
    id_abstain, id_pvals = sconfu.filter_batch(id_uncertainties)
    
    print("In-distribution test:")
    print(f"  Outlier rate: {np.mean(id_abstain):.2%}")
    print(f"  Mean p-value: {np.mean(id_pvals):.3f}")
    
    # Test on out-of-distribution samples (should have low p-values)
    n_ood = 50
    ood_uncertainties = np.random.exponential(scale=3.0, size=n_ood)  # Different scale
    
    ood_abstain, ood_pvals = sconfu.filter_batch(ood_uncertainties)
    
    print("\nOut-of-distribution test:")
    print(f"  Outlier rate: {np.mean(ood_abstain):.2%}")
    print(f"  Mean p-value: {np.mean(ood_pvals):.3f}")
    
    # Adaptive filter
    print("\n--- Adaptive Filter ---")
    adaptive_filter = create_sconfu_filter(delta=0.05, adaptive=True)
    adaptive_filter.calibrate(cal_uncertainties)
    
    # Many OOD samples
    many_ood = np.random.exponential(scale=3.0, size=200)
    ood_abstain, ood_pvals, adjusted_delta = adaptive_filter.filter_batch_adaptive(many_ood)
    
    print(f"Adjusted delta: {adjusted_delta:.4f}")
    print(f"Final outlier rate: {np.mean(ood_abstain):.2%}")

