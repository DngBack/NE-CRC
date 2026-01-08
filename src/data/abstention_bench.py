"""AbstentionBench dataset loader with human-validated judges."""

from typing import List, Dict, Optional, Any
import random
from datasets import load_dataset
from loguru import logger

from .data_types import Sample, ScenarioType, SplitType


class AbstentionBenchLoader:
    """Loader for AbstentionBench datasets with scenario filtering."""
    
    # Key datasets from AbstentionBench focusing on shift scenarios
    DATASET_CONFIGS = {
        "unknown": {
            "name": "unknown_info",
            "scenario": ScenarioType.UNKNOWN,
            "description": "Questions about unknown or very obscure information",
        },
        "false_premise": {
            "name": "false_premise",
            "scenario": ScenarioType.FALSE_PREMISE,
            "description": "Questions with incorrect assumptions",
        },
        "outdated": {
            "name": "outdated_info",
            "scenario": ScenarioType.OUTDATED,
            "description": "Questions about outdated information",
        },
        "underspecified": {
            "name": "underspecified",
            "scenario": ScenarioType.UNDERSPECIFIED,
            "description": "Ambiguous or underspecified queries",
        },
        "multi_hop": {
            "name": "multi_hop",
            "scenario": ScenarioType.MULTI_HOP,
            "description": "Multi-hop reasoning requiring multiple steps",
        },
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Initialize AbstentionBench loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
            seed: Random seed for reproducibility
        """
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
    
    def load_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
    ) -> List[Sample]:
        """Load a specific AbstentionBench dataset.
        
        Args:
            dataset_name: Name of dataset (key from DATASET_CONFIGS)
            max_samples: Maximum number of samples to load
        
        Returns:
            List of Sample objects
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.DATASET_CONFIGS.keys())}"
            )
        
        config = self.DATASET_CONFIGS[dataset_name]
        logger.info(f"Loading dataset: {dataset_name} ({config['description']})")
        
        # For now, create synthetic data structure
        # TODO: Replace with actual HuggingFace dataset loading when available
        # dataset = load_dataset("abstention-bench", config["name"], cache_dir=self.cache_dir)
        
        samples = self._load_synthetic_dataset(dataset_name, config, max_samples)
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def _load_synthetic_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[Sample]:
        """Create synthetic dataset for testing (replace with real data loader).
        
        Args:
            dataset_name: Name of dataset
            config: Dataset configuration
            max_samples: Maximum number of samples
        
        Returns:
            List of synthetic samples
        """
        # Create synthetic samples based on scenario type
        num_samples = max_samples if max_samples else 1000
        samples = []
        
        scenario = config["scenario"]
        
        for i in range(num_samples):
            # Generate synthetic query based on scenario
            if scenario == ScenarioType.UNKNOWN:
                query = f"What is the population of the fictional city Zephoria-{i}?"
                # Should abstain - unknown information
                # Mix of incorrect (0.0), correct (1.0), and unlabeled (None)
                if i % 3 == 0:
                    correctness = 0.0  # Incorrect
                elif i % 3 == 1:
                    correctness = 1.0  # Correct (some known info)
                else:
                    correctness = None  # Unlabeled
            elif scenario == ScenarioType.FALSE_PREMISE:
                query = f"When did George Washington use his iPhone to call Congress-{i}?"
                # Should abstain - false premise
                # Mix of incorrect (0.0), correct (1.0), and unlabeled (None)
                if i % 3 == 0:
                    correctness = 0.0  # Incorrect
                elif i % 3 == 1:
                    correctness = 1.0  # Correct
                else:
                    correctness = None  # Unlabeled
            elif scenario == ScenarioType.OUTDATED:
                query = f"What is the current stock price of Company-{i} (as of 2020)?"
                # Information becomes outdated
                correctness = 0.3 if i % 2 == 0 else 0.7
            elif scenario == ScenarioType.UNDERSPECIFIED:
                query = f"How tall is the building-{i}?"
                # Ambiguous - which building?
                correctness = 0.2 if i % 4 == 0 else 0.6
            elif scenario == ScenarioType.MULTI_HOP:
                query = f"If X-{i} is larger than Y-{i} and Y-{i} is larger than Z-{i}, what is the relationship between X-{i} and Z-{i}?"
                # Multi-hop reasoning
                correctness = 0.8 if i % 5 != 0 else 0.3
            else:
                query = f"Generic question {i}"
                correctness = 0.5
            
            sample = Sample(
                query=query,
                answer=None,  # To be filled by LLM
                correctness=correctness,
                sample_id=f"{dataset_name}_{i}",
                dataset_name=dataset_name,
                scenario=scenario,
                metadata={
                    "scenario": scenario.value,
                    "synthetic": True,
                }
            )
            samples.append(sample)
        
        return samples
    
    def load_multiple_datasets(
        self,
        dataset_names: List[str],
        max_samples_per_dataset: Optional[int] = None,
    ) -> Dict[str, List[Sample]]:
        """Load multiple datasets.
        
        Args:
            dataset_names: List of dataset names to load
            max_samples_per_dataset: Maximum samples per dataset
        
        Returns:
            Dictionary mapping dataset names to sample lists
        """
        datasets = {}
        for name in dataset_names:
            datasets[name] = self.load_dataset(name, max_samples_per_dataset)
        
        return datasets
    
    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset names."""
        return list(cls.DATASET_CONFIGS.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASET_CONFIGS[dataset_name]


def create_default_loader(seed: int = 42) -> AbstentionBenchLoader:
    """Create a default AbstentionBench loader.
    
    Args:
        seed: Random seed
    
    Returns:
        Configured loader
    """
    return AbstentionBenchLoader(
        cache_dir=".cache/abstention_bench",
        seed=seed,
    )


if __name__ == "__main__":
    # Example usage
    loader = create_default_loader()
    
    # Load one dataset
    samples = loader.load_dataset("unknown", max_samples=100)
    print(f"Loaded {len(samples)} samples")
    print(f"First sample: {samples[0]}")
    
    # Load multiple datasets
    datasets = loader.load_multiple_datasets(
        ["unknown", "false_premise", "outdated"],
        max_samples_per_dataset=50
    )
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, samples in datasets.items():
        print(f"  {name}: {len(samples)} samples")

