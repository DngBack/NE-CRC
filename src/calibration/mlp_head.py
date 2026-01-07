"""MLP calibration head for UniCR."""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ..data import EvidenceFeatures


class EvidenceDataset(Dataset):
    """PyTorch dataset for evidence features."""
    
    def __init__(self, features: List[EvidenceFeatures], labels: List[float]):
        """Initialize dataset.
        
        Args:
            features: List of evidence features
            labels: List of correctness labels
        """
        self.X = torch.FloatTensor([f.to_array() for f in features])
        self.y = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPCalibrationHead(nn.Module):
    """MLP calibration head for UniCR."""
    
    def __init__(
        self,
        input_dim: int = 9,  # Default number of evidence features
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        scale_features: bool = True,
    ):
        """Initialize MLP calibration head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            scale_features: Whether to standardize features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.scale_features = scale_features
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Scaler
        self.scaler = StandardScaler() if scale_features else None
        self.fitted = False
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Probabilities [batch_size, 1]
        """
        return self.network(x)
    
    def fit(
        self,
        features: List[EvidenceFeatures],
        labels: List[float],
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        val_split: float = 0.2,
        early_stopping_patience: int = 10,
    ):
        """Fit calibration head on training data.
        
        Args:
            features: List of evidence features
            labels: List of correctness labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            val_split: Validation split ratio
            early_stopping_patience: Patience for early stopping
        """
        if len(features) != len(labels):
            raise ValueError("Number of features must match number of labels")
        
        logger.info(f"Fitting MLP head on {len(features)} samples")
        
        # Scale features
        X = np.array([f.to_array() for f in features])
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Update features with scaled values
        scaled_features = []
        for i, f in enumerate(features):
            f_copy = EvidenceFeatures(sample_id=f.sample_id)
            f_copy.__dict__.update(dict(zip(EvidenceFeatures.feature_names(), X_scaled[i])))
            scaled_features.append(f_copy)
        
        # Split train/val
        n_val = int(len(scaled_features) * val_split)
        val_features = scaled_features[:n_val]
        val_labels = labels[:n_val]
        train_features = scaled_features[n_val:]
        train_labels = labels[n_val:]
        
        # Create datasets
        train_dataset = EvidenceDataset(train_features, train_labels)
        val_dataset = EvidenceDataset(val_features, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1)
                    
                    outputs = self(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Log
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.fitted = True
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    def predict_proba(
        self,
        features: List[EvidenceFeatures],
    ) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            features: List of evidence features
        
        Returns:
            Array of calibrated probabilities
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Scale features
        X = np.array([f.to_array() for f in features])
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self(X_tensor).cpu().numpy().squeeze()
        
        # Ensure 1D array
        if probs.ndim == 0:
            probs = np.array([probs])
        
        return probs
    
    def predict(
        self,
        features: List[EvidenceFeatures],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels.
        
        Args:
            features: List of evidence features
            threshold: Decision threshold
        
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(features)
        return (probs >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate,
                'scale_features': self.scale_features,
            }
        }, path)
        
        logger.info(f"Saved MLP head to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        config = checkpoint['config']
        
        self.input_dim = config['input_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout_rate']
        self.scale_features = config['scale_features']
        
        self.fitted = True
        self.eval()
        logger.info(f"Loaded MLP head from {path}")


def create_mlp_head(**kwargs) -> MLPCalibrationHead:
    """Factory function to create MLP calibration head.
    
    Args:
        **kwargs: Arguments for MLPCalibrationHead
    
    Returns:
        MLPCalibrationHead instance
    """
    return MLPCalibrationHead(**kwargs)


if __name__ == "__main__":
    # Example usage
    from ..data import EvidenceFeatures
    
    # Create synthetic training data
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 200
    
    features = []
    labels = []
    
    for i in range(n_samples):
        # Generate random features
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
        
        # Generate label correlated with consistency
        label = 1.0 if consistency > 0.5 else 0.0
        
        features.append(f)
        labels.append(label)
    
    # Train model
    head = create_mlp_head(hidden_dims=[64, 32])
    head.fit(features[:160], labels[:160], epochs=30)
    
    # Evaluate
    probs = head.predict_proba(features[160:])
    preds = head.predict(features[160:])
    
    print(f"\nTest predictions: {preds[:10]}")
    print(f"Test probabilities: {probs[:10]}")
    print(f"True labels: {labels[160:170]}")

