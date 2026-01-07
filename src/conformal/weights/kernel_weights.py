"""Kernel-based weights for NE-CRC."""

import numpy as np
from typing import Optional
from loguru import logger


class KernelWeights:
    """Compute kernel-based weights for NE-CRC."""
    
    def __init__(
        self,
        kernel: str = "rbf",
        bandwidth: Optional[float] = None,
        normalize: bool = True,
    ):
        """Initialize kernel weights.
        
        Args:
            kernel: Kernel type ("rbf", "linear", "polynomial")
            bandwidth: Bandwidth parameter for RBF kernel (auto if None)
            normalize: Whether to normalize weights to sum to 1
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        
        valid_kernels = ["rbf", "linear", "polynomial"]
        if kernel not in valid_kernels:
            raise ValueError(f"Invalid kernel: {kernel}. Choose from {valid_kernels}")
    
    def compute(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Compute kernel weights.
        
        Args:
            cal_features: Calibration features [n_cal, feature_dim]
            test_feature: Test feature [feature_dim]
        
        Returns:
            Weights [n_cal]
        """
        if cal_features is None or test_feature is None:
            n_cal = len(cal_features) if cal_features is not None else 1
            return np.ones(n_cal)
        
        # Compute kernel values
        if self.kernel == "rbf":
            weights = self._rbf_kernel(cal_features, test_feature)
        elif self.kernel == "linear":
            weights = self._linear_kernel(cal_features, test_feature)
        elif self.kernel == "polynomial":
            weights = self._polynomial_kernel(cal_features, test_feature)
        
        # Normalize
        if self.normalize:
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights
    
    def _rbf_kernel(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """RBF (Gaussian) kernel.
        
        k(x, x') = exp(-||x - x'||^2 / (2 * bandwidth^2))
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
        
        Returns:
            Kernel values [n_cal]
        """
        # Compute squared distances
        diff = cal_features - test_feature
        sq_distances = np.sum(diff ** 2, axis=1)
        
        # Auto bandwidth: median heuristic
        if self.bandwidth is None:
            bandwidth = np.median(np.sqrt(sq_distances))
            if bandwidth == 0:
                bandwidth = 1.0
        else:
            bandwidth = self.bandwidth
        
        # RBF kernel
        kernel_values = np.exp(-sq_distances / (2 * bandwidth ** 2))
        
        return kernel_values
    
    def _linear_kernel(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
    ) -> np.ndarray:
        """Linear kernel.
        
        k(x, x') = <x, x'>
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
        
        Returns:
            Kernel values [n_cal]
        """
        # Dot product
        kernel_values = np.dot(cal_features, test_feature)
        
        # Shift to positive (add minimum + 1)
        kernel_values = kernel_values - np.min(kernel_values) + 1.0
        
        return kernel_values
    
    def _polynomial_kernel(
        self,
        cal_features: np.ndarray,
        test_feature: np.ndarray,
        degree: int = 2,
        coef0: float = 1.0,
    ) -> np.ndarray:
        """Polynomial kernel.
        
        k(x, x') = (<x, x'> + coef0)^degree
        
        Args:
            cal_features: [n_cal, feature_dim]
            test_feature: [feature_dim]
            degree: Polynomial degree
            coef0: Coefficient
        
        Returns:
            Kernel values [n_cal]
        """
        # Dot product
        dot_products = np.dot(cal_features, test_feature)
        
        # Polynomial kernel
        kernel_values = (dot_products + coef0) ** degree
        
        return kernel_values
    
    def __call__(self, cal_features, test_feature):
        """Make object callable."""
        return self.compute(cal_features, test_feature)


def create_kernel_weights(
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
) -> KernelWeights:
    """Factory function to create kernel weights.
    
    Args:
        kernel: Kernel type
        bandwidth: Bandwidth for RBF kernel
    
    Returns:
        KernelWeights instance
    """
    return KernelWeights(kernel=kernel, bandwidth=bandwidth)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate random features
    n_cal = 50
    feature_dim = 10
    cal_features = np.random.randn(n_cal, feature_dim)
    test_feature = np.random.randn(feature_dim)
    
    # Test different kernels
    for kernel in ["rbf", "linear", "polynomial"]:
        weights_fn = create_kernel_weights(kernel=kernel)
        weights = weights_fn(cal_features, test_feature)
        
        print(f"\n{kernel} kernel:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights sum: {np.sum(weights):.3f}")
        print(f"  Max weight: {np.max(weights):.4f}")
        print(f"  Min weight: {np.min(weights):.4f}")
        print(f"  Top 5 weights: {weights[np.argsort(weights)[-5:]]}")

