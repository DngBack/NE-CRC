"""Semantic dispersion computation for uncertainty quantification."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class DispersionEstimator:
    """Compute semantic dispersion (variance) over multiple generations."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metric: str = "variance",
    ):
        """Initialize dispersion estimator.
        
        Args:
            model_name: Sentence transformer model for embeddings
            metric: Dispersion metric ("variance", "std", "max_dist", "mean_dist")
        """
        self.metric = metric
        
        valid_metrics = ["variance", "std", "max_dist", "mean_dist", "entropy"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Choose from {valid_metrics}")
        
        logger.info(f"Loading sentence transformer for dispersion: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def compute(
        self,
        generations: List[str],
        normalize: bool = True,
    ) -> float:
        """Compute semantic dispersion over generations.
        
        Dispersion measures the spread/variance of embeddings in semantic space.
        Higher dispersion indicates more disagreement/uncertainty.
        
        Args:
            generations: List of generated texts
            normalize: Whether to normalize answers before embedding
        
        Returns:
            Dispersion value (higher = more uncertainty)
        """
        if len(generations) <= 1:
            return 0.0
        
        # Normalize texts if requested
        if normalize:
            texts = [self._normalize_text(g) for g in generations]
        else:
            texts = generations
        
        # Encode texts to embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Compute dispersion metric
        if self.metric == "variance":
            dispersion = self._compute_variance(embeddings)
        elif self.metric == "std":
            dispersion = self._compute_std(embeddings)
        elif self.metric == "max_dist":
            dispersion = self._compute_max_distance(embeddings)
        elif self.metric == "mean_dist":
            dispersion = self._compute_mean_distance(embeddings)
        elif self.metric == "entropy":
            dispersion = self._compute_embedding_entropy(embeddings)
        
        return float(dispersion)
    
    def compute_batch(
        self,
        generations_list: List[List[str]],
        normalize: bool = True,
    ) -> List[float]:
        """Compute dispersion for a batch of generation sets.
        
        Args:
            generations_list: List of generation lists
            normalize: Whether to normalize answers
        
        Returns:
            List of dispersion values
        """
        logger.info(f"Computing dispersion for {len(generations_list)} samples")
        return [self.compute(gens, normalize) for gens in generations_list]
    
    def _compute_variance(self, embeddings: np.ndarray) -> float:
        """Compute variance of embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Total variance (sum of variances across dimensions)
        """
        # Compute variance along each dimension, then sum
        variances = np.var(embeddings, axis=0)
        return float(np.sum(variances))
    
    def _compute_std(self, embeddings: np.ndarray) -> float:
        """Compute standard deviation (sqrt of variance).
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Standard deviation
        """
        return float(np.sqrt(self._compute_variance(embeddings)))
    
    def _compute_max_distance(self, embeddings: np.ndarray) -> float:
        """Compute maximum pairwise distance.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Maximum cosine distance between any two embeddings
        """
        from scipy.spatial.distance import pdist
        
        # Compute pairwise cosine distances
        distances = pdist(embeddings, metric='cosine')
        
        if len(distances) == 0:
            return 0.0
        
        return float(np.max(distances))
    
    def _compute_mean_distance(self, embeddings: np.ndarray) -> float:
        """Compute mean pairwise distance.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Mean cosine distance between embeddings
        """
        from scipy.spatial.distance import pdist
        
        # Compute pairwise cosine distances
        distances = pdist(embeddings, metric='cosine')
        
        if len(distances) == 0:
            return 0.0
        
        return float(np.mean(distances))
    
    def _compute_embedding_entropy(self, embeddings: np.ndarray) -> float:
        """Compute entropy-like measure from embedding distribution.
        
        This discretizes the embedding space and computes entropy.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Entropy-like dispersion measure
        """
        # Use PCA to reduce to 1D
        from sklearn.decomposition import PCA
        
        if len(embeddings) < 2:
            return 0.0
        
        pca = PCA(n_components=1)
        reduced = pca.fit_transform(embeddings)
        
        # Discretize into bins
        hist, _ = np.histogram(reduced, bins=10, density=True)
        
        # Normalize to probabilities
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        
        # Compute entropy
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log(p)
        
        return float(entropy)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        import re
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


def create_dispersion_estimator(
    metric: str = "variance",
    **kwargs,
) -> DispersionEstimator:
    """Factory function to create dispersion estimator.
    
    Args:
        metric: Dispersion metric
        **kwargs: Additional arguments
    
    Returns:
        DispersionEstimator instance
    """
    return DispersionEstimator(metric=metric, **kwargs)


if __name__ == "__main__":
    # Example usage
    estimator = create_dispersion_estimator(metric="variance")
    
    # Test generations with varying similarity
    generations = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The answer is Paris.",
        "I don't know the answer.",
        "The capital of Germany is Berlin.",
    ]
    
    dispersion = estimator.compute(generations)
    print(f"Dispersion (variance): {dispersion:.3f}")
    
    # Test different metrics
    for metric in ["std", "max_dist", "mean_dist"]:
        est = create_dispersion_estimator(metric=metric)
        disp = est.compute(generations)
        print(f"Dispersion ({metric}): {disp:.3f}")

