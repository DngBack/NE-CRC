"""Evidence extraction from LLM outputs for calibration."""

import re
from typing import List, Optional, Dict
import numpy as np
from collections import Counter
from loguru import logger

from ..data import GeneratedOutput, EvidenceFeatures


class EvidenceExtractor:
    """Extract evidence features from LLM generations."""
    
    def __init__(self):
        """Initialize evidence extractor."""
        # Patterns for verbal confidence extraction
        self.confidence_patterns = [
            (r"I am (\d+)% confident", lambda m: float(m.group(1)) / 100),
            (r"confidence[:\s]+(\d+)%", lambda m: float(m.group(1)) / 100),
            (r"I'm (very |quite |somewhat )?(?:confident|certain|sure)", self._parse_verbal_modifier),
            (r"(definitely|certainly|surely|absolutely)", lambda m: 0.9),
            (r"(probably|likely|presumably)", lambda m: 0.7),
            (r"(possibly|perhaps|maybe|might)", lambda m: 0.5),
            (r"(unlikely|doubtful|uncertain)", lambda m: 0.3),
        ]
    
    def extract_features(
        self,
        generated_output: GeneratedOutput,
        sample_id: str,
    ) -> EvidenceFeatures:
        """Extract all evidence features from generated output.
        
        Args:
            generated_output: LLM generated output with K samples
            sample_id: Unique identifier for this sample
        
        Returns:
            EvidenceFeatures object
        """
        generations = generated_output.generations
        token_logprobs = generated_output.token_logprobs
        
        # Consistency-based features
        consistency_score = self._compute_consistency(generations)
        
        # Token-level features
        if token_logprobs:
            token_entropy = self._compute_token_entropy(token_logprobs)
            max_token_prob = self._compute_max_token_prob(token_logprobs)
            mean_token_prob = self._compute_mean_token_prob(token_logprobs)
        else:
            token_entropy = 0.0
            max_token_prob = 0.0
            mean_token_prob = 0.0
        
        # Verbal confidence
        verbal_confidence = self._extract_verbal_confidence(generations)
        
        # Generation statistics
        generation_length = int(np.mean([len(g.split()) for g in generations]))
        num_unique_answers = self._count_unique_answers(generations)
        
        # Note: semantic_entropy and dispersion will be computed separately
        # by the uncertainty module since they require embeddings
        
        return EvidenceFeatures(
            sample_id=sample_id,
            consistency_score=consistency_score,
            semantic_entropy=0.0,  # To be filled by uncertainty module
            token_entropy=token_entropy,
            max_token_prob=max_token_prob,
            mean_token_prob=mean_token_prob,
            dispersion=0.0,  # To be filled by uncertainty module
            verbal_confidence=verbal_confidence,
            generation_length=generation_length,
            num_unique_answers=num_unique_answers,
        )
    
    def extract_batch(
        self,
        generated_outputs: List[GeneratedOutput],
        sample_ids: List[str],
    ) -> List[EvidenceFeatures]:
        """Extract features for a batch of outputs.
        
        Args:
            generated_outputs: List of generated outputs
            sample_ids: List of sample IDs
        
        Returns:
            List of EvidenceFeatures
        """
        if len(generated_outputs) != len(sample_ids):
            raise ValueError("Number of outputs must match number of sample IDs")
        
        logger.info(f"Extracting evidence features for {len(generated_outputs)} samples")
        
        features = []
        for output, sample_id in zip(generated_outputs, sample_ids):
            feature = self.extract_features(output, sample_id)
            features.append(feature)
        
        return features
    
    def _compute_consistency(self, generations: List[str]) -> float:
        """Compute self-consistency score (agreement among generations).
        
        Args:
            generations: List of generated texts
        
        Returns:
            Consistency score in [0, 1]
        """
        if len(generations) <= 1:
            return 1.0
        
        # Normalize texts for comparison
        normalized = [self._normalize_answer(g) for g in generations]
        
        # Count most common answer
        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        
        # Consistency = fraction agreeing with most common
        consistency = most_common_count / len(generations)
        return float(consistency)
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Take first sentence/clause (often the direct answer)
        text = text.split('.')[0]
        text = text.split(',')[0]
        
        return text[:100]  # Limit length
    
    def _compute_token_entropy(self, token_logprobs: List[List[float]]) -> float:
        """Compute average token entropy across samples.
        
        Args:
            token_logprobs: List of token logprobs for each sample
        
        Returns:
            Average token entropy
        """
        if not token_logprobs:
            return 0.0
        
        entropies = []
        for sample_logprobs in token_logprobs:
            if sample_logprobs:
                # Convert logprobs to probs
                probs = np.exp(sample_logprobs)
                # Clip to avoid log(0)
                probs = np.clip(probs, 1e-10, 1.0)
                # Compute entropy: -sum(p * log(p))
                entropy = -np.sum(probs * np.log(probs))
                entropies.append(entropy)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    def _compute_max_token_prob(self, token_logprobs: List[List[float]]) -> float:
        """Compute maximum token probability across samples.
        
        Args:
            token_logprobs: List of token logprobs for each sample
        
        Returns:
            Maximum token probability
        """
        if not token_logprobs:
            return 0.0
        
        max_probs = []
        for sample_logprobs in token_logprobs:
            if sample_logprobs:
                probs = np.exp(sample_logprobs)
                max_probs.append(np.max(probs))
        
        return float(np.max(max_probs)) if max_probs else 0.0
    
    def _compute_mean_token_prob(self, token_logprobs: List[List[float]]) -> float:
        """Compute mean token probability across samples.
        
        Args:
            token_logprobs: List of token logprobs for each sample
        
        Returns:
            Mean token probability
        """
        if not token_logprobs:
            return 0.0
        
        all_probs = []
        for sample_logprobs in token_logprobs:
            if sample_logprobs:
                probs = np.exp(sample_logprobs)
                all_probs.extend(probs)
        
        return float(np.mean(all_probs)) if all_probs else 0.0
    
    def _extract_verbal_confidence(self, generations: List[str]) -> float:
        """Extract verbal confidence from generated text.
        
        Args:
            generations: List of generated texts
        
        Returns:
            Average confidence score in [0, 1]
        """
        confidences = []
        
        for text in generations:
            text_lower = text.lower()
            
            # Try each pattern
            for pattern, extractor in self.confidence_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        conf = extractor(match)
                        if conf is not None:
                            confidences.append(conf)
                            break  # Use first match
                    except:
                        continue
        
        # If no explicit confidence found, return neutral
        if not confidences:
            return 0.5
        
        return float(np.mean(confidences))
    
    def _parse_verbal_modifier(self, match) -> Optional[float]:
        """Parse verbal confidence modifiers.
        
        Args:
            match: Regex match object
        
        Returns:
            Confidence score or None
        """
        text = match.group(0).lower()
        
        if 'very' in text:
            return 0.9
        elif 'quite' in text:
            return 0.8
        elif 'somewhat' in text:
            return 0.6
        else:
            return 0.7  # Default for "confident/certain/sure"
    
    def _count_unique_answers(self, generations: List[str]) -> int:
        """Count number of unique normalized answers.
        
        Args:
            generations: List of generated texts
        
        Returns:
            Number of unique answers
        """
        normalized = [self._normalize_answer(g) for g in generations]
        return len(set(normalized))
    
    def select_best_answer(
        self,
        generated_output: GeneratedOutput,
        method: str = "majority_vote",
    ) -> str:
        """Select the best answer from multiple generations.
        
        Args:
            generated_output: Generated output with K samples
            method: Selection method ("majority_vote", "first", "longest")
        
        Returns:
            Selected answer
        """
        generations = generated_output.generations
        
        if not generations:
            return ""
        
        if method == "majority_vote":
            # Return most common normalized answer
            normalized = [self._normalize_answer(g) for g in generations]
            counter = Counter(normalized)
            most_common_norm = counter.most_common(1)[0][0]
            
            # Find first generation matching most common
            for gen, norm in zip(generations, normalized):
                if norm == most_common_norm:
                    return gen
            
            return generations[0]
        
        elif method == "first":
            return generations[0]
        
        elif method == "longest":
            return max(generations, key=len)
        
        else:
            raise ValueError(f"Unknown selection method: {method}")


def create_evidence_extractor() -> EvidenceExtractor:
    """Factory function to create evidence extractor.
    
    Returns:
        EvidenceExtractor instance
    """
    return EvidenceExtractor()


if __name__ == "__main__":
    # Example usage
    from ..data import GeneratedOutput
    
    # Create synthetic output
    output = GeneratedOutput(
        query="What is 2+2?",
        generations=[
            "The answer is 4. I am 95% confident.",
            "2 plus 2 equals 4.",
            "Four (4) is the answer.",
        ],
        token_logprobs=[
            [-0.1, -0.05, -0.02, -0.01],
            [-0.15, -0.08, -0.03],
            [-0.12, -0.06, -0.04, -0.02],
        ],
        total_tokens=20,
    )
    
    # Extract features
    extractor = create_evidence_extractor()
    features = extractor.extract_features(output, "test_001")
    
    print(f"\nExtracted features:")
    print(f"  Consistency: {features.consistency_score:.3f}")
    print(f"  Token entropy: {features.token_entropy:.3f}")
    print(f"  Max token prob: {features.max_token_prob:.3f}")
    print(f"  Mean token prob: {features.mean_token_prob:.3f}")
    print(f"  Verbal confidence: {features.verbal_confidence:.3f}")
    print(f"  Generation length: {features.generation_length}")
    print(f"  Unique answers: {features.num_unique_answers}")
    
    # Select best answer
    best = extractor.select_best_answer(output)
    print(f"\nBest answer: {best}")

