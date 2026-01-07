"""Train/calibration/test split generation with distribution shift scenarios."""

import random
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from loguru import logger

from .data_types import Sample, DataSplit, ShiftType, SplitType, ScenarioType


class ShiftSplitter:
    """Generate train/calibration/test splits with various shift scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize splitter with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def create_split(
        self,
        samples: List[Sample],
        shift_type: ShiftType,
        split_ratios: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        dataset_name: str = "",
    ) -> DataSplit:
        """Create train/calibration/test split with specified shift type.
        
        Args:
            samples: List of samples to split
            shift_type: Type of distribution shift
            split_ratios: (train, calibration, test) ratios
            dataset_name: Name of the dataset
        
        Returns:
            DataSplit with labeled samples
        """
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        if shift_type == ShiftType.ID:
            return self._create_id_split(samples, split_ratios, dataset_name)
        elif shift_type == ShiftType.MILD:
            return self._create_mild_shift_split(samples, split_ratios, dataset_name)
        elif shift_type == ShiftType.STRONG:
            return self._create_strong_shift_split(samples, split_ratios, dataset_name)
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")
    
    def _create_id_split(
        self,
        samples: List[Sample],
        split_ratios: Tuple[float, float, float],
        dataset_name: str,
    ) -> DataSplit:
        """Create in-distribution split (random shuffle)."""
        logger.info(f"Creating ID split with ratios {split_ratios}")
        
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split by ratios
        n = len(shuffled)
        train_end = int(n * split_ratios[0])
        cal_end = train_end + int(n * split_ratios[1])
        
        train = shuffled[:train_end]
        calibration = shuffled[train_end:cal_end]
        test = shuffled[cal_end:]
        
        # Label splits
        for s in train:
            s.split = SplitType.TRAIN
        for s in calibration:
            s.split = SplitType.CALIBRATION
        for s in test:
            s.split = SplitType.TEST
        
        return DataSplit(
            train=train,
            calibration=calibration,
            test=test,
            shift_type=ShiftType.ID,
            dataset_name=dataset_name,
            split_params={"method": "random", "ratios": split_ratios},
        )
    
    def _create_mild_shift_split(
        self,
        samples: List[Sample],
        split_ratios: Tuple[float, float, float],
        dataset_name: str,
    ) -> DataSplit:
        """Create mild shift split (cross-topic/scenario).
        
        Strategy: Train/cal on easier samples, test on harder samples.
        Proxy for difficulty: use correctness score as difficulty indicator.
        """
        logger.info(f"Creating MILD shift split with ratios {split_ratios}")
        
        # Sort by correctness (easier to harder)
        # Samples with higher correctness are "easier"
        sorted_samples = sorted(
            samples,
            key=lambda s: s.correctness if s.correctness is not None else 0.5,
            reverse=True,
        )
        
        # Train/cal on easier half, test on harder half
        n = len(sorted_samples)
        easy_end = int(n * 0.6)  # 60% easier samples for train+cal
        
        easy_samples = sorted_samples[:easy_end]
        hard_samples = sorted_samples[easy_end:]
        
        # Split easy samples for train and calibration
        random.shuffle(easy_samples)
        train_end = int(len(easy_samples) * split_ratios[0] / (split_ratios[0] + split_ratios[1]))
        
        train = easy_samples[:train_end]
        calibration = easy_samples[train_end:]
        test = hard_samples
        
        # Label splits
        for s in train:
            s.split = SplitType.TRAIN
        for s in calibration:
            s.split = SplitType.CALIBRATION
        for s in test:
            s.split = SplitType.TEST
        
        return DataSplit(
            train=train,
            calibration=calibration,
            test=test,
            shift_type=ShiftType.MILD,
            dataset_name=dataset_name,
            split_params={
                "method": "difficulty_based",
                "easy_ratio": 0.6,
                "ratios": split_ratios
            },
        )
    
    def _create_strong_shift_split(
        self,
        samples: List[Sample],
        split_ratios: Tuple[float, float, float],
        dataset_name: str,
    ) -> DataSplit:
        """Create strong shift split (temporal/domain shift).
        
        Strategy: Partition by scenario type if available.
        Train/cal on "normal" scenarios, test on "shifted" scenarios (outdated, false_premise).
        """
        logger.info(f"Creating STRONG shift split with ratios {split_ratios}")
        
        # Partition by scenario
        shifted_scenarios = {ScenarioType.OUTDATED, ScenarioType.FALSE_PREMISE}
        
        shifted_samples = [s for s in samples if s.scenario in shifted_scenarios]
        normal_samples = [s for s in samples if s.scenario not in shifted_scenarios]
        
        if not shifted_samples:
            logger.warning("No shifted scenarios found, falling back to mild shift")
            return self._create_mild_shift_split(samples, split_ratios, dataset_name)
        
        if not normal_samples:
            logger.warning("No normal scenarios found, falling back to ID split")
            return self._create_id_split(samples, split_ratios, dataset_name)
        
        # Train and calibration from normal scenarios
        random.shuffle(normal_samples)
        train_end = int(len(normal_samples) * split_ratios[0] / (split_ratios[0] + split_ratios[1]))
        
        train = normal_samples[:train_end]
        calibration = normal_samples[train_end:]
        
        # Test from shifted scenarios
        test = shifted_samples
        
        # Balance if needed
        if len(test) > len(train) + len(calibration):
            # Too many test samples, move some to calibration
            random.shuffle(test)
            excess = len(test) - len(train)
            calibration.extend(test[excess:])
            test = test[:excess]
        
        # Label splits
        for s in train:
            s.split = SplitType.TRAIN
        for s in calibration:
            s.split = SplitType.CALIBRATION
        for s in test:
            s.split = SplitType.TEST
        
        return DataSplit(
            train=train,
            calibration=calibration,
            test=test,
            shift_type=ShiftType.STRONG,
            dataset_name=dataset_name,
            split_params={
                "method": "scenario_based",
                "shifted_scenarios": [s.value for s in shifted_scenarios],
                "ratios": split_ratios,
            },
        )
    
    def create_multi_dataset_split(
        self,
        dataset_samples: Dict[str, List[Sample]],
        shift_type: ShiftType,
        split_ratios: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> DataSplit:
        """Create split across multiple datasets.
        
        Args:
            dataset_samples: Dictionary mapping dataset names to sample lists
            shift_type: Type of distribution shift
            split_ratios: (train, calibration, test) ratios
        
        Returns:
            Combined DataSplit
        """
        logger.info(f"Creating multi-dataset split with {len(dataset_samples)} datasets")
        
        all_train = []
        all_calibration = []
        all_test = []
        
        for dataset_name, samples in dataset_samples.items():
            split = self.create_split(samples, shift_type, split_ratios, dataset_name)
            all_train.extend(split.train)
            all_calibration.extend(split.calibration)
            all_test.extend(split.test)
        
        # Shuffle combined splits
        random.shuffle(all_train)
        random.shuffle(all_calibration)
        random.shuffle(all_test)
        
        return DataSplit(
            train=all_train,
            calibration=all_calibration,
            test=all_test,
            shift_type=shift_type,
            dataset_name="multi_dataset",
            split_params={
                "datasets": list(dataset_samples.keys()),
                "num_datasets": len(dataset_samples),
                "ratios": split_ratios,
            },
        )


def create_default_splitter(seed: int = 42) -> ShiftSplitter:
    """Create a default shift splitter.
    
    Args:
        seed: Random seed
    
    Returns:
        Configured splitter
    """
    return ShiftSplitter(seed=seed)


if __name__ == "__main__":
    # Example usage
    from .abstention_bench import create_default_loader
    
    # Load data
    loader = create_default_loader()
    samples = loader.load_dataset("unknown", max_samples=100)
    
    # Create different shift splits
    splitter = create_default_splitter()
    
    # ID split
    id_split = splitter.create_split(samples, ShiftType.ID, dataset_name="unknown")
    print(f"\nID split:")
    print(f"  Train: {len(id_split.train)}")
    print(f"  Cal: {len(id_split.calibration)}")
    print(f"  Test: {len(id_split.test)}")
    
    # Mild shift
    mild_split = splitter.create_split(samples, ShiftType.MILD, dataset_name="unknown")
    print(f"\nMILD shift split:")
    print(f"  Train: {len(mild_split.train)}")
    print(f"  Cal: {len(mild_split.calibration)}")
    print(f"  Test: {len(mild_split.test)}")
    
    # Strong shift
    strong_split = splitter.create_split(samples, ShiftType.STRONG, dataset_name="unknown")
    print(f"\nSTRONG shift split:")
    print(f"  Train: {len(strong_split.train)}")
    print(f"  Cal: {len(strong_split.calibration)}")
    print(f"  Test: {len(strong_split.test)}")

