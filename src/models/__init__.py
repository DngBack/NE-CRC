"""LLM inference and evidence extraction module."""

from .llm_wrapper import (
    BaseLLMInference,
    VLLMInference,
    TransformersInference,
    create_llm_inference,
    VLLM_AVAILABLE,
)
from .evidence_extractor import (
    EvidenceExtractor,
    create_evidence_extractor,
)

__all__ = [
    # LLM inference
    "BaseLLMInference",
    "VLLMInference",
    "TransformersInference",
    "create_llm_inference",
    "VLLM_AVAILABLE",
    # Evidence extraction
    "EvidenceExtractor",
    "create_evidence_extractor",
]

