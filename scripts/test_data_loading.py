#!/usr/bin/env python3
"""Test script to verify data loading and correctness evaluation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_default_loader, create_correctness_evaluator
from src.models import create_llm_inference, create_evidence_extractor
from loguru import logger
import numpy as np


def test_data_loading():
    """Test loading real dataset from HuggingFace."""
    logger.info("=" * 80)
    logger.info("Test 1: Loading AbstentionBench dataset")
    logger.info("=" * 80)
    
    loader = create_default_loader()
    
    # Try loading with real data
    try:
        samples = loader.load_dataset("unknown", max_samples=10, use_real_data=True)
        logger.info(f"✓ Successfully loaded {len(samples)} samples")
        
        if samples:
            sample = samples[0]
            logger.info(f"\nSample example:")
            logger.info(f"  Query: {sample.query[:100]}...")
            logger.info(f"  Metadata keys: {list(sample.metadata.keys())}")
            logger.info(f"  Has reference_answers: {'reference_answers' in sample.metadata}")
            logger.info(f"  Has should_abstain: {'should_abstain' in sample.metadata}")
            logger.info(f"  Is synthetic: {sample.metadata.get('synthetic', True)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load real dataset: {e}")
        logger.info("This is expected if dataset is not available or network issues")
        return False


def test_correctness_evaluator():
    """Test correctness evaluator."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Correctness Evaluator")
    logger.info("=" * 80)
    
    evaluator = create_correctness_evaluator(method="exact")
    
    # Test case 1: Correct answer
    correctness = evaluator.evaluate(
        llm_answer="Paris",
        reference_answers=["Paris", "The capital of France is Paris"],
        should_abstain=False,
        model_abstained=False,
    )
    logger.info(f"Test 1 (correct answer): {correctness} (expected: 1.0)")
    assert correctness == 1.0, "Test 1 failed"
    
    # Test case 2: Should abstain and did abstain
    correctness = evaluator.evaluate(
        llm_answer="I don't know",
        reference_answers=[],
        should_abstain=True,
        model_abstained=True,
    )
    logger.info(f"Test 2 (correct abstention): {correctness} (expected: 1.0)")
    assert correctness == 1.0, "Test 2 failed"
    
    # Test case 3: Should abstain but answered
    correctness = evaluator.evaluate(
        llm_answer="Some answer",
        reference_answers=[],
        should_abstain=True,
        model_abstained=False,
    )
    logger.info(f"Test 3 (incorrect: answered when should abstain): {correctness} (expected: 0.0)")
    assert correctness == 0.0, "Test 3 failed"
    
    # Test case 4: Wrong answer
    correctness = evaluator.evaluate(
        llm_answer="London",
        reference_answers=["Paris"],
        should_abstain=False,
        model_abstained=False,
    )
    logger.info(f"Test 4 (wrong answer): {correctness} (expected: 0.0)")
    assert correctness == 0.0, "Test 4 failed"
    
    logger.info("✓ All correctness evaluator tests passed")
    return True


def test_end_to_end_small():
    """Test end-to-end with small dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: End-to-end pipeline (small dataset)")
    logger.info("=" * 80)
    
    try:
        from src.data import ExperimentConfig, ShiftType
        from src.pipeline import create_experiment_pipeline
        
        # Create minimal config
        config = ExperimentConfig(
            dataset_names=["unknown"],
            shift_type=ShiftType.ID,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            num_samples=3,  # Small for testing
            calibration_head="logistic",
            alpha=0.05,
            delta=0.05,
            uncertainty_method="semantic_entropy",
            weight_method="similarity",
            experiment_name="test_pipeline",
            seed=42,
        )
        
        logger.info("Creating pipeline...")
        pipeline = create_experiment_pipeline(config)
        
        logger.info("Running pipeline (this may take a while for LLM inference)...")
        logger.info("Note: This will use cached generations if available")
        
        # Run pipeline
        results, metrics = pipeline.run()
        
        logger.info("\n✓ Pipeline completed successfully!")
        logger.info(f"Results: {len(results)} systems evaluated")
        logger.info(f"Metrics computed for: {list(metrics.keys())}")
        
        # Check metrics are reasonable
        for system_name, system_metrics in metrics.items():
            coverage = system_metrics.get('coverage', 0)
            risk = system_metrics.get('selective_risk', 0)
            logger.info(f"  {system_name}: coverage={coverage:.3f}, risk={risk:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("Starting data loading and correctness evaluation tests...\n")
    
    results = []
    
    # Test 1: Data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test 2: Correctness evaluator
    results.append(("Correctness Evaluator", test_correctness_evaluator()))
    
    # Test 3: End-to-end (optional, can be slow)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full end-to-end test")
    args = parser.parse_args()
    
    if args.full:
        results.append(("End-to-End Pipeline", test_end_to_end_small()))
    else:
        logger.info("\nSkipping end-to-end test (use --full to run)")
        logger.info("Note: End-to-end test requires LLM inference and may take time")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

