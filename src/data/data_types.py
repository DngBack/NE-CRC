"""Data types for the NE-CRC project."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class SplitType(Enum):
    """Data split types."""
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class ShiftType(Enum):
    """Types of distribution shift."""
    ID = "id"  # In-distribution (random split)
    MILD = "mild"  # Cross-topic/domain
    STRONG = "strong"  # Temporal + domain shift


class ScenarioType(Enum):
    """AbstentionBench scenario types."""
    UNKNOWN = "unknown"
    FALSE_PREMISE = "false_premise"
    OUTDATED = "outdated"
    UNDERSPECIFIED = "underspecified"
    SUBJECTIVE = "subjective"
    MULTI_HOP = "multi_hop"


@dataclass
class Sample:
    """Single data sample with query, answer, and correctness label."""
    
    query: str
    answer: Optional[str] = None
    correctness: Optional[float] = None  # 0.0 (wrong) to 1.0 (correct), or None if unlabeled
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional fields for experiments
    sample_id: Optional[str] = None
    dataset_name: Optional[str] = None
    scenario: Optional[ScenarioType] = None
    split: Optional[SplitType] = None
    
    def __post_init__(self):
        """Validate sample data."""
        if self.correctness is not None:
            if not 0.0 <= self.correctness <= 1.0:
                raise ValueError(f"Correctness must be in [0, 1], got {self.correctness}")


@dataclass
class GeneratedOutput:
    """LLM generated output with multiple samples and metadata."""
    
    query: str
    generations: List[str]  # K sampled generations
    token_logprobs: Optional[List[List[float]]] = None  # Per-token log probabilities
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Computed features
    best_answer: Optional[str] = None  # Selected answer (e.g., majority vote)
    consistency_score: Optional[float] = None  # Self-consistency agreement
    verbal_confidence: Optional[float] = None  # Parsed confidence from text
    
    # Cost tracking
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass
class EvidenceFeatures:
    """Extracted evidence features for calibration."""
    
    sample_id: str
    
    # Consistency-based features
    consistency_score: float = 0.0  # Agreement among K samples
    semantic_entropy: float = 0.0  # Cluster-based entropy
    
    # Token-level features
    token_entropy: float = 0.0  # Average per-token entropy
    max_token_prob: float = 0.0  # Maximum token probability
    mean_token_prob: float = 0.0  # Mean token probability
    
    # Semantic features
    dispersion: float = 0.0  # Variance in embedding space
    verbal_confidence: float = 0.5  # Parsed confidence from text
    
    # Additional features
    generation_length: int = 0  # Average generation length
    num_unique_answers: int = 0  # Number of unique answers
    
    def to_array(self) -> List[float]:
        """Convert features to array for ML models."""
        return [
            self.consistency_score,
            self.semantic_entropy,
            self.token_entropy,
            self.max_token_prob,
            self.mean_token_prob,
            self.dispersion,
            self.verbal_confidence,
            float(self.generation_length),
            float(self.num_unique_answers),
        ]
    
    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names in same order as to_array()."""
        return [
            "consistency_score",
            "semantic_entropy",
            "token_entropy",
            "max_token_prob",
            "mean_token_prob",
            "dispersion",
            "verbal_confidence",
            "generation_length",
            "num_unique_answers",
        ]


@dataclass
class DataSplit:
    """Train/calibration/test split with metadata."""
    
    train: List[Sample]
    calibration: List[Sample]
    test: List[Sample]
    
    shift_type: ShiftType = ShiftType.ID
    dataset_name: str = ""
    split_params: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        """Total number of samples."""
        return len(self.train) + len(self.calibration) + len(self.test)
    
    def get_split(self, split_type: SplitType) -> List[Sample]:
        """Get samples for a specific split."""
        if split_type == SplitType.TRAIN:
            return self.train
        elif split_type == SplitType.CALIBRATION:
            return self.calibration
        elif split_type == SplitType.TEST:
            return self.test
        else:
            raise ValueError(f"Unknown split type: {split_type}")


@dataclass
class PredictionResult:
    """Prediction result with decision and metadata."""
    
    sample_id: str
    query: str
    
    # Model outputs
    answer: Optional[str] = None
    confidence: float = 0.0  # c(x) from calibration head
    uncertainty: float = 0.0  # u(x) for filtering
    
    # Decision
    decision: str = "abstain"  # "answer" or "abstain"
    decision_reason: str = ""  # Why this decision was made
    
    # Ground truth (if available)
    correctness: Optional[float] = None
    
    # Threshold values (for analysis)
    threshold: Optional[float] = None  # τ(x) from CRC/NE-CRC
    p_value: Optional[float] = None  # p-value from SConU filter
    
    # Cost
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    
    def is_answered(self) -> bool:
        """Check if sample was answered (not abstained)."""
        return self.decision == "answer"
    
    def is_correct(self) -> Optional[bool]:
        """Check if answered sample is correct (None if abstained or no ground truth)."""
        if not self.is_answered() or self.correctness is None:
            return None
        return self.correctness >= 0.5  # Binary threshold


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    
    # Data
    dataset_names: List[str]
    shift_type: ShiftType = ShiftType.ID
    split_ratios: tuple = (0.4, 0.3, 0.3)  # train/cal/test
    
    # Model
    model_name: str = "meta-llama/Llama-3-8B"
    num_samples: int = 10  # K for semantic entropy
    
    # Calibration
    calibration_head: str = "logistic"  # "logistic" or "mlp"
    
    # Conformal risk control
    alpha: float = 0.05  # Risk level
    delta: float = 0.05  # Outlier threshold for SConU
    
    # Uncertainty & weights
    uncertainty_method: str = "semantic_entropy"  # or "token_entropy", "dispersion"
    weight_method: str = "similarity"  # or "density_ratio", "kernel"
    
    # System variant
    system: str = "s_unicr"  # "unicr", "unicr_filter", "unicr_necrc", "s_unicr"
    
    # Experiment tracking
    experiment_name: str = ""
    seed: int = 42
    cache_dir: str = ".cache"
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Set experiment name if not provided."""
        if not self.experiment_name:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.system}_{self.shift_type.value}_{timestamp}"

