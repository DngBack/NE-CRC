"""Logistic regression calibration head for UniCR."""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger
import pickle

from ..data import EvidenceFeatures


class LogisticCalibrationHead:
    """Logistic regression head for calibrating confidence scores."""
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        scale_features: bool = True,
    ):
        """Initialize logistic calibration head.
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations for solver
            class_weight: Weights for classes ('balanced' or None)
            scale_features: Whether to standardize features
        """
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.scale_features = scale_features
        
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
        )
        
        self.scaler = StandardScaler() if scale_features else None
        self.fitted = False
    
    def fit(
        self,
        features: List[EvidenceFeatures],
        labels: List[float],
    ):
        """Fit calibration head on training data.
        
        Args:
            features: List of evidence features
            labels: List of correctness labels (0 or 1)
        """
        if len(features) != len(labels):
            raise ValueError("Number of features must match number of labels")
        
        logger.info(f"Fitting logistic head on {len(features)} samples")
        
        # Convert features to array
        X = np.array([f.to_array() for f in features])
        y = np.array(labels)
        
        # Convert continuous labels to binary if needed
        y_binary = (y >= 0.5).astype(int)
        
        # Check if we have at least 2 classes
        unique_classes = np.unique(y_binary)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Need samples from at least 2 classes for training, "
                f"but only found class(es): {unique_classes.tolist()}. "
                f"Please ensure your dataset has both correct and incorrect samples."
            )
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X, y_binary)
        self.fitted = True
        
        # Log training performance
        train_acc = self.model.score(X, y_binary)
        logger.info(f"Training accuracy: {train_acc:.3f}")
    
    def predict_proba(
        self,
        features: List[EvidenceFeatures],
    ) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            features: List of evidence features
        
        Returns:
            Array of calibrated probabilities (p(correct))
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert features to array
        X = np.array([f.to_array() for f in features])
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict probabilities ([:, 1] gives p(class=1))
        probs = self.model.predict_proba(X)[:, 1]
        
        return probs
    
    def predict(
        self,
        features: List[EvidenceFeatures],
    ) -> np.ndarray:
        """Predict binary labels.
        
        Args:
            features: List of evidence features
        
        Returns:
            Array of binary predictions
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert features to array
        X = np.array([f.to_array() for f in features])
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_feature_importances(self) -> dict:
        """Get feature importances (coefficients).
        
        Returns:
            Dictionary mapping feature names to coefficients
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")
        
        coeffs = self.model.coef_[0]
        feature_names = EvidenceFeatures.feature_names()
        
        return dict(zip(feature_names, coeffs))
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': {
                    'C': self.C,
                    'max_iter': self.max_iter,
                    'class_weight': self.class_weight,
                    'scale_features': self.scale_features,
                }
            }, f)
        
        logger.info(f"Saved logistic head to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        config = data['config']
        
        self.C = config['C']
        self.max_iter = config['max_iter']
        self.class_weight = config['class_weight']
        self.scale_features = config['scale_features']
        
        self.fitted = True
        logger.info(f"Loaded logistic head from {path}")


def create_logistic_head(**kwargs) -> LogisticCalibrationHead:
    """Factory function to create logistic calibration head.
    
    Args:
        **kwargs: Arguments for LogisticCalibrationHead
    
    Returns:
        LogisticCalibrationHead instance
    """
    return LogisticCalibrationHead(**kwargs)


if __name__ == "__main__":
    # Example usage
    from ..data import EvidenceFeatures
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 100
    
    features = []
    labels = []
    
    for i in range(n_samples):
        # Generate random features
        f = EvidenceFeatures(
            sample_id=f"sample_{i}",
            consistency_score=np.random.rand(),
            semantic_entropy=np.random.rand() * 2,
            token_entropy=np.random.rand(),
            max_token_prob=0.5 + np.random.rand() * 0.5,
            mean_token_prob=0.5 + np.random.rand() * 0.3,
            dispersion=np.random.rand() * 3,
            verbal_confidence=np.random.rand(),
            generation_length=10 + int(np.random.rand() * 20),
            num_unique_answers=1 + int(np.random.rand() * 5),
        )
        
        # Generate label correlated with consistency
        label = 1 if f.consistency_score > 0.5 else 0
        
        features.append(f)
        labels.append(label)
    
    # Train model
    head = create_logistic_head()
    head.fit(features[:80], labels[:80])
    
    # Evaluate
    probs = head.predict_proba(features[80:])
    preds = head.predict(features[80:])
    
    print(f"Test predictions: {preds[:10]}")
    print(f"Test probabilities: {probs[:10]}")
    
    # Feature importances
    importances = head.get_feature_importances()
    print("\nFeature importances:")
    for name, coef in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.3f}")

