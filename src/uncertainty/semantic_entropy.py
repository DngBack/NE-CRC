"""Semantic entropy computation for uncertainty quantification."""

from typing import List, Optional
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from loguru import logger


class SemanticEntropyEstimator:
    """Compute semantic entropy over multiple generations."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clustering_threshold: float = 0.5,
        use_exact_match: bool = False,
    ):
        """Initialize semantic entropy estimator.
        
        Args:
            model_name: Sentence transformer model for embeddings
            clustering_threshold: Similarity threshold for clustering
            use_exact_match: Whether to use exact string matching instead of embeddings
        """
        self.use_exact_match = use_exact_match
        self.clustering_threshold = clustering_threshold
        
        if not use_exact_match:
            logger.info(f"Loading sentence transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
            logger.info("Using exact string matching for semantic entropy")
    
    def compute(
        self,
        generations: List[str],
        normalize: bool = True,
    ) -> float:
        """Compute semantic entropy over generations.
        
        Semantic entropy clusters semantically similar answers and computes
        entropy over the cluster distribution.
        
        Args:
            generations: List of generated texts
            normalize: Whether to normalize answers before clustering
        
        Returns:
            Semantic entropy value (higher = more uncertainty)
        """
        if len(generations) <= 1:
            return 0.0
        
        # Normalize texts if requested
        if normalize:
            texts = [self._normalize_text(g) for g in generations]
        else:
            texts = generations
        
        # Cluster texts
        cluster_labels = self._cluster_texts(texts)
        
        # Compute cluster probabilities
        cluster_counts = Counter(cluster_labels)
        total = len(cluster_labels)
        cluster_probs = [count / total for count in cluster_counts.values()]
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = 0.0
        for p in cluster_probs:
            if p > 0:
                entropy -= p * np.log(p)
        
        return float(entropy)
    
    def compute_batch(
        self,
        generations_list: List[List[str]],
        normalize: bool = True,
    ) -> List[float]:
        """Compute semantic entropy for a batch of generation sets.
        
        Args:
            generations_list: List of generation lists
            normalize: Whether to normalize answers
        
        Returns:
            List of semantic entropy values
        """
        logger.info(f"Computing semantic entropy for {len(generations_list)} samples")
        return [self.compute(gens, normalize) for gens in generations_list]
    
    def _cluster_texts(self, texts: List[str]) -> List[int]:
        """Cluster texts based on semantic similarity.
        
        Args:
            texts: List of texts to cluster
        
        Returns:
            List of cluster labels
        """
        if self.use_exact_match:
            return self._cluster_exact_match(texts)
        else:
            return self._cluster_embeddings(texts)
    
    def _cluster_exact_match(self, texts: List[str]) -> List[int]:
        """Cluster by exact string matching.
        
        Args:
            texts: List of texts
        
        Returns:
            Cluster labels
        """
        unique_texts = {}
        labels = []
        
        for text in texts:
            if text not in unique_texts:
                unique_texts[text] = len(unique_texts)
            labels.append(unique_texts[text])
        
        return labels
    
    def _cluster_embeddings(self, texts: List[str]) -> List[int]:
        """Cluster using semantic embeddings.
        
        Args:
            texts: List of texts
        
        Returns:
            Cluster labels
        """
        # Encode texts
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Cluster using DBSCAN
        # Convert similarity threshold to distance threshold
        eps = 1.0 - self.clustering_threshold
        
        clusterer = DBSCAN(eps=eps, min_samples=1, metric='cosine')
        labels = clusterer.fit_predict(embeddings)
        
        # Handle noise points (label -1) by giving each a unique cluster
        max_label = max(labels) if len(labels) > 0 else -1
        labels = [
            label if label != -1 else max_label + 1 + i
            for i, label in enumerate(labels)
        ]
        
        return labels
    
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
        
        # Take first sentence/clause (often the direct answer)
        text = text.split('.')[0]
        
        return text
    
    def get_cluster_info(
        self,
        generations: List[str],
        normalize: bool = True,
    ) -> dict:
        """Get detailed clustering information.
        
        Args:
            generations: List of generated texts
            normalize: Whether to normalize answers
        
        Returns:
            Dictionary with cluster information
        """
        if normalize:
            texts = [self._normalize_text(g) for g in generations]
        else:
            texts = generations
        
        labels = self._cluster_texts(texts)
        
        # Group texts by cluster
        clusters = {}
        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
        
        # Compute cluster probabilities
        cluster_probs = {
            label: len(members) / len(texts)
            for label, members in clusters.items()
        }
        
        return {
            "num_clusters": len(clusters),
            "cluster_sizes": {label: len(members) for label, members in clusters.items()},
            "cluster_probs": cluster_probs,
            "clusters": clusters,
        }


def create_semantic_entropy_estimator(
    use_exact_match: bool = False,
    **kwargs,
) -> SemanticEntropyEstimator:
    """Factory function to create semantic entropy estimator.
    
    Args:
        use_exact_match: Whether to use exact matching
        **kwargs: Additional arguments
    
    Returns:
        SemanticEntropyEstimator instance
    """
    return SemanticEntropyEstimator(use_exact_match=use_exact_match, **kwargs)


if __name__ == "__main__":
    # Example usage
    estimator = create_semantic_entropy_estimator(use_exact_match=False)
    
    # Test generations
    generations = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The answer is Paris.",
        "I don't know the answer.",
        "I'm not sure, maybe Paris?",
    ]
    
    entropy = estimator.compute(generations)
    print(f"Semantic entropy: {entropy:.3f}")
    
    # Get cluster info
    info = estimator.get_cluster_info(generations)
    print(f"\nNumber of clusters: {info['num_clusters']}")
    print(f"Cluster sizes: {info['cluster_sizes']}")
    print(f"Cluster probabilities: {info['cluster_probs']}")

