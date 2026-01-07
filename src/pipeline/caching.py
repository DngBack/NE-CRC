"""Caching utilities for expensive LLM inference."""

import pickle
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
from loguru import logger

from ..data import GeneratedOutput, EvidenceFeatures


class CacheManager:
    """Manage caching of LLM outputs and features."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.generations_dir = self.cache_dir / "generations"
        self.features_dir = self.cache_dir / "features"
        self.results_dir = self.cache_dir / "results"
        
        self.generations_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for cache key.
        
        Args:
            data: Data to hash
        
        Returns:
            Hash string
        """
        # Convert to string and hash
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_generations_cache_path(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ) -> Path:
        """Get cache path for generations.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name
            split: Split name (train/cal/test)
            config: Generation config
        
        Returns:
            Cache file path
        """
        config_hash = self._compute_hash(config)
        filename = f"{model_name.replace('/', '_')}_{dataset_name}_{split}_{config_hash}.pkl"
        return self.generations_dir / filename
    
    def save_generations(
        self,
        generations: List[GeneratedOutput],
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ):
        """Save generations to cache.
        
        Args:
            generations: List of generated outputs
            model_name: Model name
            dataset_name: Dataset name
            split: Split name
            config: Generation config
        """
        cache_path = self.get_generations_cache_path(model_name, dataset_name, split, config)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(generations, f)
        
        logger.info(f"Saved {len(generations)} generations to {cache_path}")
    
    def load_generations(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ) -> Optional[List[GeneratedOutput]]:
        """Load generations from cache.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name
            split: Split name
            config: Generation config
        
        Returns:
            List of generated outputs or None if not cached
        """
        cache_path = self.get_generations_cache_path(model_name, dataset_name, split, config)
        
        if not cache_path.exists():
            logger.info(f"Cache miss: {cache_path}")
            return None
        
        with open(cache_path, 'rb') as f:
            generations = pickle.load(f)
        
        logger.info(f"Loaded {len(generations)} generations from cache")
        return generations
    
    def get_features_cache_path(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ) -> Path:
        """Get cache path for features.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name
            split: Split name
            config: Feature extraction config
        
        Returns:
            Cache file path
        """
        config_hash = self._compute_hash(config)
        filename = f"features_{model_name.replace('/', '_')}_{dataset_name}_{split}_{config_hash}.pkl"
        return self.features_dir / filename
    
    def save_features(
        self,
        features: List[EvidenceFeatures],
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ):
        """Save features to cache.
        
        Args:
            features: List of evidence features
            model_name: Model name
            dataset_name: Dataset name
            split: Split name
            config: Feature extraction config
        """
        cache_path = self.get_features_cache_path(model_name, dataset_name, split, config)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        logger.info(f"Saved {len(features)} features to {cache_path}")
    
    def load_features(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        config: Dict,
    ) -> Optional[List[EvidenceFeatures]]:
        """Load features from cache.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name
            split: Split name
            config: Feature extraction config
        
        Returns:
            List of features or None if not cached
        """
        cache_path = self.get_features_cache_path(model_name, dataset_name, split, config)
        
        if not cache_path.exists():
            logger.info(f"Cache miss: {cache_path}")
            return None
        
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        
        logger.info(f"Loaded {len(features)} features from cache")
        return features
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear cache.
        
        Args:
            cache_type: Type of cache to clear ("generations", "features", "results", "all")
        """
        if cache_type in ["generations", "all"]:
            for file in self.generations_dir.glob("*.pkl"):
                file.unlink()
            logger.info("Cleared generations cache")
        
        if cache_type in ["features", "all"]:
            for file in self.features_dir.glob("*.pkl"):
                file.unlink()
            logger.info("Cleared features cache")
        
        if cache_type in ["results", "all"]:
            for file in self.results_dir.glob("*.pkl"):
                file.unlink()
            logger.info("Cleared results cache")


def create_cache_manager(cache_dir: str = ".cache") -> CacheManager:
    """Factory function to create cache manager.
    
    Args:
        cache_dir: Cache directory
    
    Returns:
        CacheManager instance
    """
    return CacheManager(cache_dir=cache_dir)


if __name__ == "__main__":
    # Example usage
    cache = create_cache_manager()
    
    # Mock data
    from ..data import GeneratedOutput
    generations = [
        GeneratedOutput(
            query=f"Query {i}",
            generations=[f"Answer {i}-{j}" for j in range(3)],
            total_tokens=100,
        )
        for i in range(10)
    ]
    
    # Save
    config = {"num_samples": 3, "temperature": 1.0}
    cache.save_generations(generations, "test-model", "test-dataset", "test", config)
    
    # Load
    loaded = cache.load_generations("test-model", "test-dataset", "test", config)
    print(f"Loaded {len(loaded)} generations")
    
    # Clear
    cache.clear_cache("generations")

