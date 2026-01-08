"""Correctness evaluator for comparing LLM outputs with reference answers."""

from typing import List, Optional, Union
import re
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, semantic similarity disabled")


class CorrectnessEvaluator:
    """Evaluate correctness of LLM outputs against reference answers."""
    
    def __init__(
        self,
        method: str = "exact",
        similarity_threshold: float = 0.8,
        use_semantic_similarity: bool = False,
    ):
        """Initialize correctness evaluator.
        
        Args:
            method: Evaluation method ("exact", "fuzzy", "semantic")
            similarity_threshold: Threshold for semantic similarity (0-1)
            use_semantic_similarity: Whether to use semantic similarity as fallback
        """
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.use_semantic_similarity = use_semantic_similarity
        
        if use_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Loading sentence transformer for semantic similarity")
            self.similarity_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.similarity_model = None
            if use_semantic_similarity:
                logger.warning("Semantic similarity requested but not available")
    
    def evaluate(
        self,
        llm_answer: str,
        reference_answers: List[str],
        should_abstain: bool = False,
        model_abstained: bool = False,
    ) -> Optional[float]:
        """Evaluate correctness of a single LLM answer.
        
        Args:
            llm_answer: Generated answer from LLM (or empty if abstained)
            reference_answers: List of reference/correct answers
            should_abstain: Whether the model should have abstained
            model_abstained: Whether the model actually abstained
        
        Returns:
            Correctness score: 1.0 (correct), 0.0 (incorrect), or None (cannot determine)
        """
        # Case 1: Model should abstain
        if should_abstain:
            if model_abstained:
                # Correct: abstained when should abstain
                return 1.0
            else:
                # Incorrect: answered when should abstain
                return 0.0
        
        # Case 2: Model should answer (should_abstain=False)
        if model_abstained:
            # Incorrect: abstained when should answer (missed opportunity)
            return 0.0
        
        # Case 3: Model answered, check if answer is correct
        if not reference_answers:
            # No reference to compare against
            # If should_abstain was False, we assume it's correct (no way to verify)
            # But this is uncertain, so return None
            return None
        
        # Compare LLM answer with references
        if self.method == "exact":
            is_correct = self._exact_match(llm_answer, reference_answers)
        elif self.method == "fuzzy":
            is_correct = self._fuzzy_match(llm_answer, reference_answers)
        elif self.method == "semantic":
            is_correct = self._semantic_match(llm_answer, reference_answers)
        else:
            # Try exact first, then semantic if available
            is_correct = self._exact_match(llm_answer, reference_answers)
            if not is_correct and self.use_semantic_similarity:
                is_correct = self._semantic_match(llm_answer, reference_answers)
        
        return 1.0 if is_correct else 0.0
    
    def evaluate_batch(
        self,
        llm_answers: List[str],
        reference_answers_list: List[List[str]],
        should_abstain_list: List[bool],
        model_abstained_list: List[bool],
    ) -> List[Optional[float]]:
        """Evaluate correctness for a batch of answers.
        
        Args:
            llm_answers: List of LLM generated answers
            reference_answers_list: List of reference answer lists
            should_abstain_list: List of should_abstain flags
            model_abstained_list: List of model_abstained flags
        
        Returns:
            List of correctness scores
        """
        if len(llm_answers) != len(reference_answers_list):
            raise ValueError("Length mismatch between answers and references")
        
        results = []
        for i, (answer, refs, should_abstain, abstained) in enumerate(
            zip(llm_answers, reference_answers_list, should_abstain_list, model_abstained_list)
        ):
            correctness = self.evaluate(answer, refs, should_abstain, abstained)
            results.append(correctness)
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _exact_match(self, answer: str, references: List[str]) -> bool:
        """Check if answer exactly matches any reference.
        
        Args:
            answer: LLM answer
            references: List of reference answers
        
        Returns:
            True if matches any reference
        """
        answer_norm = self._normalize_text(answer)
        
        for ref in references:
            ref_norm = self._normalize_text(ref)
            if answer_norm == ref_norm:
                return True
        
        return False
    
    def _fuzzy_match(self, answer: str, references: List[str]) -> bool:
        """Check if answer fuzzy matches any reference.
        
        Uses normalized text and checks if answer is contained in reference or vice versa.
        
        Args:
            answer: LLM answer
            references: List of reference answers
        
        Returns:
            True if fuzzy matches any reference
        """
        answer_norm = self._normalize_text(answer)
        
        if not answer_norm:
            return False
        
        for ref in references:
            ref_norm = self._normalize_text(ref)
            
            # Check if answer is contained in reference or vice versa
            if answer_norm in ref_norm or ref_norm in answer_norm:
                return True
            
            # Check word overlap (at least 50% of words match)
            answer_words = set(answer_norm.split())
            ref_words = set(ref_norm.split())
            
            if answer_words and ref_words:
                overlap = len(answer_words & ref_words)
                min_len = min(len(answer_words), len(ref_words))
                if min_len > 0 and overlap / min_len >= 0.5:
                    return True
        
        return False
    
    def _semantic_match(self, answer: str, references: List[str]) -> bool:
        """Check if answer semantically matches any reference.
        
        Uses sentence transformers to compute semantic similarity.
        
        Args:
            answer: LLM answer
            references: List of reference answers
        
        Returns:
            True if semantic similarity exceeds threshold
        """
        if not self.similarity_model:
            return False
        
        if not answer or not references:
            return False
        
        # Encode answer
        answer_embedding = self.similarity_model.encode([answer], show_progress_bar=False)[0]
        
        # Encode references
        ref_embeddings = self.similarity_model.encode(references, show_progress_bar=False)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(
            answer_embedding.reshape(1, -1),
            ref_embeddings
        )[0]
        
        # Check if any similarity exceeds threshold
        max_similarity = float(np.max(similarities))
        return max_similarity >= self.similarity_threshold


def create_correctness_evaluator(
    method: str = "exact",
    use_semantic_similarity: bool = False,
    **kwargs,
) -> CorrectnessEvaluator:
    """Factory function to create correctness evaluator.
    
    Args:
        method: Evaluation method
        use_semantic_similarity: Whether to use semantic similarity
        **kwargs: Additional arguments
    
    Returns:
        CorrectnessEvaluator instance
    """
    return CorrectnessEvaluator(
        method=method,
        use_semantic_similarity=use_semantic_similarity,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    evaluator = create_correctness_evaluator(method="exact")
    
    # Test cases
    test_cases = [
        {
            "llm_answer": "Paris",
            "reference_answers": ["Paris", "The capital of France is Paris"],
            "should_abstain": False,
            "model_abstained": False,
        },
        {
            "llm_answer": "I don't know",
            "reference_answers": [],
            "should_abstain": True,
            "model_abstained": True,
        },
        {
            "llm_answer": "London",
            "reference_answers": ["Paris"],
            "should_abstain": False,
            "model_abstained": False,
        },
    ]
    
    for i, case in enumerate(test_cases):
        correctness = evaluator.evaluate(**case)
        print(f"Test {i+1}: correctness = {correctness}")

