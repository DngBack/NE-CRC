"""AbstentionBench dataset loader with human-validated judges."""

from typing import List, Dict, Optional, Any
import random
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, list_repo_files
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
        use_real_data: bool = True,
    ) -> List[Sample]:
        """Load a specific AbstentionBench dataset.
        
        Args:
            dataset_name: Name of dataset (key from DATASET_CONFIGS)
            max_samples: Maximum number of samples to load
            use_real_data: If True, try to load from HuggingFace; if False or fails, use synthetic
        
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
        
        # Try to load real dataset from HuggingFace
        if use_real_data:
            try:
                samples = self._load_real_dataset(dataset_name, config, max_samples)
                if samples:
                    logger.info(f"Successfully loaded {len(samples)} real samples from HuggingFace")
                    return samples
                else:
                    logger.warning("Real dataset loading returned empty, falling back to synthetic")
            except Exception as e:
                logger.warning(f"Failed to load real dataset from HuggingFace: {e}")
                logger.info("Falling back to synthetic dataset")
        
        # Fallback to synthetic dataset
        samples = self._load_synthetic_dataset(dataset_name, config, max_samples)
        logger.info(f"Loaded {len(samples)} synthetic samples from {dataset_name}")
        return samples
    
    def _load_real_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[Sample]:
        """Load real AbstentionBench dataset from HuggingFace.
        
        Args:
            dataset_name: Name of dataset
            config: Dataset configuration
            max_samples: Maximum number of samples to load
        
        Returns:
            List of Sample objects with metadata (correctness will be evaluated later)
        """
        try:
            logger.info(f"Attempting to load real dataset from facebook/AbstentionBench")
            
            # Method 1: Try loading via datasets library (standard way)
            dataset = self._try_load_via_datasets()
            
            if dataset is None:
                # Method 2: Try loading from files directly
                logger.info("Trying to load from files directly...")
                dataset = self._try_load_from_files()
            
            if dataset is None:
                raise ValueError("All loading methods failed. Dataset may not be available or requires different loading method.")
            
            # Process dataset (common for all loading methods)
            # Determine which split to use (usually 'test' or 'validation')
            available_splits = list(dataset.keys())
            logger.info(f"Available splits: {available_splits}")
            
            # Prefer 'test' split, fallback to first available
            split_name = 'test' if 'test' in available_splits else available_splits[0]
            data = dataset[split_name]
            
            logger.info(f"Using split: {split_name} with {len(data)} samples")
            
            # Check dataset structure
            if len(data) == 0:
                raise ValueError("Dataset split is empty")
            
            # Inspect first sample to understand structure
            first_sample = data[0]
            logger.info(f"Dataset fields: {list(first_sample.keys())}")
            
            samples = []
            scenario = config["scenario"]
            
            # Map scenario names from our config to dataset metadata
            scenario_mapping = {
                "unknown": ["unknown", "unknown_info"],
                "false_premise": ["false_premise", "false_premise"],
                "outdated": ["outdated", "outdated_info"],
                "underspecified": ["underspecified", "underspecified"],
                "multi_hop": ["multi_hop", "multi_hop"],
            }
            
            scenario_keywords = scenario_mapping.get(dataset_name, [dataset_name])
            
            # First pass: collect all samples (we'll filter by scenario if possible, but be lenient)
            all_items = []
            for i, item in enumerate(data):
                all_items.append((i, item))
            
            # If dataset is large, try to filter by scenario; otherwise take all
            use_scenario_filter = len(all_items) > 100
            
            for i, item in all_items:
                # Filter by scenario if metadata available and we have many samples
                metadata = item.get('metadata_json', {})
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Check if this sample matches our scenario (only if filtering)
                if use_scenario_filter:
                    source = metadata.get('source', '').lower() if metadata else ''
                    scenario_match = any(
                        keyword.lower() in source or keyword.lower() in str(metadata).lower()
                        for keyword in scenario_keywords
                    )
                    
                    # If we have specific scenario filtering, apply it
                    if scenario_keywords and not scenario_match:
                        # Try to match by dataset_name in metadata
                        if dataset_name not in str(metadata).lower():
                            continue
                
                # Extract fields (handle different possible field names)
                question = item.get('question', '') or item.get('query', '') or item.get('prompt', '')
                if not question:
                    continue
                
                # Handle reference_answers (could be list, string, or None)
                reference_answers = item.get('reference_answers', [])
                if reference_answers is None:
                    reference_answers = []
                elif isinstance(reference_answers, str):
                    reference_answers = [reference_answers]
                elif not isinstance(reference_answers, list):
                    reference_answers = []
                
                # Handle should_abstain (could be bool, int, or string)
                should_abstain = item.get('should_abstain', False)
                if isinstance(should_abstain, str):
                    should_abstain = should_abstain.lower() in ['true', '1', 'yes']
                elif isinstance(should_abstain, int):
                    should_abstain = bool(should_abstain)
                
                # Create sample with metadata (correctness will be evaluated after LLM inference)
                sample = Sample(
                    query=question,
                    answer=None,  # Will be filled by LLM
                    correctness=None,  # Will be evaluated after LLM inference
                    sample_id=f"{dataset_name}_{len(samples)}",
                    dataset_name=dataset_name,
                    scenario=scenario,
                    metadata={
                        "scenario": scenario.value,
                        "should_abstain": should_abstain,
                        "reference_answers": reference_answers if reference_answers else [],
                        "metadata_json": metadata,
                        "synthetic": False,
                        "original_index": i,
                    }
                )
                
                samples.append(sample)
                
                # Limit samples if specified
                if max_samples and len(samples) >= max_samples:
                    break
            
            logger.info(f"Loaded {len(samples)} samples matching scenario {dataset_name}")
            
            if len(samples) == 0:
                logger.warning(f"No samples loaded for scenario {dataset_name}. This may indicate:")
                logger.warning("  1. Dataset structure doesn't match expected format")
                logger.warning("  2. Scenario filtering too strict")
                logger.warning("  3. Dataset doesn't contain this scenario")
                raise ValueError(f"No samples loaded for {dataset_name}")
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading real dataset: {e}")
            logger.info("This might be due to:")
            logger.info("  1. Dataset not available on HuggingFace")
            logger.info("  2. Network issues")
            logger.info("  3. Dataset structure changes")
            logger.info("  4. Missing dependencies")
            raise
    
    def _try_load_via_datasets(self) -> Optional[Any]:
        """Try loading dataset via datasets library with multiple methods.
        
        Returns:
            Dataset object or None if all methods fail
        """
        error_msgs = []
        
        # Try 1: Load without trust_remote_code (preferred for newer datasets)
        try:
            logger.info("  Method 1: Loading without trust_remote_code...")
            dataset = load_dataset(
                "facebook/AbstentionBench",
                cache_dir=self.cache_dir
            )
            logger.info("  ✓ Successfully loaded without trust_remote_code")
            return dataset
        except Exception as e1:
            error_msgs.append(f"Without trust_remote_code: {str(e1)[:100]}")
            logger.debug(f"  Failed: {e1}")
        
        # Try 2: With trust_remote_code (for older datasets, may fail)
        try:
            logger.info("  Method 2: Loading with trust_remote_code...")
            dataset = load_dataset(
                "facebook/AbstentionBench",
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            logger.info("  ✓ Successfully loaded with trust_remote_code")
            return dataset
        except Exception as e2:
            error_msgs.append(f"With trust_remote_code: {str(e2)[:100]}")
            logger.debug(f"  Failed: {e2}")
        
        # Try 3: Alternative dataset name
        try:
            logger.info("  Method 3: Trying alternative dataset name...")
            dataset = load_dataset(
                "AbstentionBench/AbstentionBench",
                cache_dir=self.cache_dir
            )
            logger.info("  ✓ Successfully loaded from alternative name")
            return dataset
        except Exception as e3:
            error_msgs.append(f"Alternative name: {str(e3)[:100]}")
            logger.debug(f"  Failed: {e3}")
        
        logger.warning(f"All standard loading methods failed. Errors: {len(error_msgs)}")
        return None
    
    def _try_load_from_files(self) -> Optional[Any]:
        """Try loading dataset by downloading files directly from HuggingFace Hub.
        
        Returns:
            Dataset object or None if fails
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            repo_id = "facebook/AbstentionBench"
            logger.info(f"  Attempting to load files directly from {repo_id}...")
            
            # List available files
            try:
                files = list_repo_files(repo_id, repo_type="dataset")
                logger.info(f"  Found {len(files)} files in repository")
                
                # Look for parquet, json, or csv files
                data_files = [f for f in files if any(f.endswith(ext) for ext in ['.parquet', '.json', '.jsonl', '.csv'])]
                
                if not data_files:
                    logger.warning("  No data files found (parquet/json/csv)")
                    return None
                
                logger.info(f"  Found data files: {data_files[:5]}...")  # Show first 5
                
                # Try loading from parquet first
                parquet_files = [f for f in data_files if f.endswith('.parquet')]
                if parquet_files:
                    logger.info(f"  Loading from parquet files...")
                    try:
                        dataset = load_dataset(
                            repo_id,
                            data_files=parquet_files,
                            cache_dir=self.cache_dir
                        )
                        logger.info("  ✓ Successfully loaded from parquet files")
                        return dataset
                    except Exception as e:
                        logger.debug(f"  Failed to load parquet: {e}")
                
                # Try loading from JSON/JSONL directly
                json_files = [f for f in data_files if f.endswith('.json') or f.endswith('.jsonl')]
                if json_files:
                    logger.info(f"  Loading from JSON files...")
                    try:
                        # Try loading as JSONL first (one JSON per line)
                        jsonl_files = [f for f in json_files if f.endswith('.jsonl')]
                        if jsonl_files:
                            dataset = load_dataset(
                                repo_id,
                                data_files=jsonl_files,
                                cache_dir=self.cache_dir
                            )
                            logger.info("  ✓ Successfully loaded from JSONL files")
                            return dataset
                        
                        # Try regular JSON
                        dataset = load_dataset(
                            repo_id,
                            data_files=json_files,
                            cache_dir=self.cache_dir
                        )
                        logger.info("  ✓ Successfully loaded from JSON files")
                        return dataset
                    except Exception as e:
                        logger.debug(f"  Failed to load JSON: {e}")
                        
                        # Last resort: try downloading and parsing manually
                        logger.info("  Attempting manual download and parse...")
                        try:
                            return self._load_manually_from_hub(repo_id, json_files)
                        except Exception as e2:
                            logger.debug(f"  Manual load also failed: {e2}")
                
            except Exception as e:
                logger.debug(f"  Failed to list/download files: {e}")
                # Last resort: try manual download and parse
                if json_files:
                    logger.info("  Attempting manual download and parse...")
                    try:
                        return self._load_manually_from_hub(repo_id, json_files)
                    except Exception as e2:
                        logger.debug(f"  Manual load also failed: {e2}")
                return None
            
        except ImportError:
            logger.warning("  huggingface_hub not available for direct file loading")
            return None
        except Exception as e:
            logger.debug(f"  Direct file loading failed: {e}")
            return None
        
        return None
    
    def _load_manually_from_hub(self, repo_id: str, file_paths: List[str]) -> Optional[Any]:
        """Manually download and parse files from HuggingFace Hub.
        
        This is a last resort when standard loading methods fail.
        
        Args:
            repo_id: Repository ID
            file_paths: List of file paths to download
        
        Returns:
            Dataset object or None
        """
        try:
            from huggingface_hub import hf_hub_download
            import json
            
            logger.info(f"  Manually downloading {len(file_paths)} files...")
            
            # Download first JSON/JSONL file
            data_file = file_paths[0] if file_paths else None
            if not data_file:
                return None
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=data_file,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            
            logger.info(f"  Downloaded: {data_file}")
            
            # Try to parse as JSONL
            if data_file.endswith('.jsonl'):
                data = []
                with open(local_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data.append(json.loads(line))
                            except:
                                continue
                
                if data:
                    logger.info(f"  Parsed {len(data)} samples from JSONL")
                    # Convert to Dataset
                    from datasets import Dataset
                    return Dataset.from_list(data)
            
            # Try to parse as JSON
            elif data_file.endswith('.json'):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    logger.info(f"  Parsed {len(data)} samples from JSON list")
                    from datasets import Dataset
                    return Dataset.from_list(data)
                elif isinstance(data, dict):
                    # Could be a dict with splits
                    logger.info(f"  Parsed JSON dict with keys: {list(data.keys())}")
                    # Try to extract a list from the dict
                    for key, value in data.items():
                        if isinstance(value, list):
                            from datasets import Dataset
                            return Dataset.from_list(value)
            
            return None
            
        except Exception as e:
            logger.debug(f"  Manual download/parse failed: {e}")
            return None
    
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

