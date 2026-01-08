"""End-to-end experiment pipeline."""

import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from ..data import (
    ExperimentConfig,
    create_default_loader,
    create_default_splitter,
    ShiftType,
    SplitType,
    create_correctness_evaluator,
)
from ..models import create_llm_inference, create_evidence_extractor
from ..uncertainty import create_uncertainty_estimator
from ..conformal.weights import create_similarity_weights
from ..baselines import create_unicr_baseline, create_heuristic_baseline
from ..variants import create_unicr_with_filter, create_unicr_with_necrc, create_s_unicr
from ..metrics import create_metrics_computer
from .caching import create_cache_manager


class ExperimentPipeline:
    """End-to-end experiment pipeline."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize pipeline.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Initialize components
        self.loader = create_default_loader(seed=config.seed)
        self.splitter = create_default_splitter(seed=config.seed)
        self.cache = create_cache_manager(cache_dir=config.cache_dir)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized experiment: {config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run(self):
        """Run complete experiment pipeline."""
        logger.info("=" * 80)
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 1. Load and split data
        logger.info("\n[1/8] Loading and splitting data...")
        data_split = self._load_and_split_data()
        
        # 2. Run LLM inference (with caching)
        logger.info("\n[2/9] Running LLM inference...")
        generations = self._run_llm_inference(data_split)
        
        # 3. Evaluate correctness from LLM outputs
        logger.info("\n[3/9] Evaluating correctness from LLM outputs...")
        data_split = self._evaluate_correctness(generations, data_split)
        
        # 4. Extract evidence features
        logger.info("\n[4/9] Extracting evidence features...")
        features = self._extract_features(generations, data_split)
        
        # 5. Compute uncertainties
        logger.info("\n[5/9] Computing uncertainties...")
        uncertainties = self._compute_uncertainties(generations)
        
        # 6. Prepare weight features for NE-CRC
        logger.info("\n[6/9] Preparing weight features...")
        weight_features = self._prepare_weight_features(features)
        
        # 7. Run all systems
        logger.info("\n[7/9] Running all system variants...")
        results, test_labels_filtered = self._run_all_systems(
            features, uncertainties, weight_features, data_split
        )
        
        # 8. Evaluate metrics
        logger.info("\n[8/9] Evaluating metrics...")
        metrics = self._evaluate_metrics(results, test_labels_filtered)
        
        # 9. Save results
        logger.info("\n[9/9] Saving results...")
        self._save_results(results, metrics)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Experiment completed in {elapsed:.2f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'=' * 80}\n")
        
        return results, metrics
    
    def _load_and_split_data(self):
        """Load dataset and create splits."""
        # Load dataset
        dataset_name = self.config.dataset_names[0]  # Use first dataset
        samples = self.loader.load_dataset(dataset_name)
        
        # Validate loaded data
        logger.info(f"Validating loaded dataset...")
        if not samples:
            raise ValueError(f"No samples loaded from dataset: {dataset_name}")
        
        logger.info(f"Loaded {len(samples)} total samples")
        
        # Check for metadata (indicates real vs synthetic data)
        has_metadata = any(s.metadata.get('synthetic', True) == False for s in samples)
        if has_metadata:
            logger.info("✓ Real dataset detected (has metadata)")
        else:
            logger.warning("⚠ Using synthetic dataset (no real metadata found)")
        
        # Create split
        shift_type = self.config.shift_type
        data_split = self.splitter.create_split(
            samples,
            shift_type,
            self.config.split_ratios,
            dataset_name
        )
        
        # Validate splits
        logger.info(f"\nDataset: {dataset_name}")
        logger.info(f"Shift type: {shift_type.value}")
        logger.info(f"Train: {len(data_split.train)} samples")
        logger.info(f"Calibration: {len(data_split.calibration)} samples")
        logger.info(f"Test: {len(data_split.test)} samples")
        
        # Check for empty splits
        if len(data_split.train) == 0:
            raise ValueError("Train split is empty!")
        if len(data_split.calibration) == 0:
            raise ValueError("Calibration split is empty!")
        if len(data_split.test) == 0:
            raise ValueError("Test split is empty!")
        
        # Check minimum sizes
        min_cal_size = 10
        if len(data_split.calibration) < min_cal_size:
            logger.warning(
                f"Calibration set is very small ({len(data_split.calibration)} samples). "
                f"Recommended: at least {min_cal_size} samples for reliable CRC."
            )
        
        return data_split
    
    def _run_llm_inference(self, data_split):
        """Run LLM inference with caching."""
        gen_config = {
            "num_samples": self.config.num_samples,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        
        # Try loading from cache
        dataset_name = self.config.dataset_names[0]
        
        # Check cache for all splits
        cached_generations = {}
        for split_type in [SplitType.TRAIN, SplitType.CALIBRATION, SplitType.TEST]:
            split_name = split_type.value
            cached = self.cache.load_generations(
                self.config.model_name,
                dataset_name,
                split_name,
                gen_config
            )
            if cached:
                cached_generations[split_name] = cached
        
        # If all cached, return
        if len(cached_generations) == 3:
            logger.info("All generations loaded from cache")
            return cached_generations
        
        # Otherwise, run inference
        logger.info("Running LLM inference (this may take a while)...")
        llm = create_llm_inference(self.config.model_name, use_vllm=True)
        
        generations = {}
        for split_type in [SplitType.TRAIN, SplitType.CALIBRATION, SplitType.TEST]:
            split_name = split_type.value
            
            if split_name in cached_generations:
                generations[split_name] = cached_generations[split_name]
                continue
            
            samples = data_split.get_split(split_type)
            queries = [s.query for s in samples]
            
            logger.info(f"Generating for {split_name} split ({len(queries)} samples)...")
            gens = llm.generate(
                queries,
                num_samples=self.config.num_samples,
                max_tokens=512,
                temperature=1.0,
                return_logprobs=True,
            )
            
            generations[split_name] = gens
            
            # Cache
            self.cache.save_generations(
                gens, self.config.model_name, dataset_name, split_name, gen_config
            )
        
        return generations
    
    def _evaluate_correctness(self, generations, data_split):
        """Evaluate correctness from LLM outputs and update samples.
        
        Args:
            generations: Dictionary of generated outputs per split
            data_split: DataSplit with samples
        
        Returns:
            Updated DataSplit with correctness labels
        """
        logger.info("Evaluating correctness from LLM outputs...")
        
        # Create correctness evaluator
        evaluator = create_correctness_evaluator(method="exact", use_semantic_similarity=False)
        
        # Create evidence extractor to get best answers
        extractor = create_evidence_extractor()
        
        # Process each split
        for split_type in [SplitType.TRAIN, SplitType.CALIBRATION, SplitType.TEST]:
            split_name = split_type.value
            samples = data_split.get_split(split_type)
            gens = generations[split_name]
            
            logger.info(f"Evaluating correctness for {split_name} split ({len(samples)} samples)...")
            
            # Extract best answers from generations
            llm_answers = []
            reference_answers_list = []
            should_abstain_list = []
            model_abstained_list = []
            
            for sample, gen_output in zip(samples, gens):
                # Get best answer (majority vote or first)
                best_answer = extractor.select_best_answer(gen_output, method="majority_vote")
                llm_answers.append(best_answer)
                
                # Get metadata
                metadata = sample.metadata
                reference_answers = metadata.get('reference_answers', [])
                if not reference_answers:
                    reference_answers = []
                should_abstain = metadata.get('should_abstain', False)
                
                reference_answers_list.append(reference_answers)
                should_abstain_list.append(should_abstain)
                
                # Check if model abstained (answer is empty or contains abstention phrases)
                abstention_phrases = [
                    "i don't know", "i cannot", "i'm not sure", "i don't have",
                    "unable to", "cannot answer", "no information", "not available"
                ]
                answer_lower = best_answer.lower().strip()
                model_abstained = (
                    not best_answer or
                    len(best_answer.strip()) == 0 or
                    any(phrase in answer_lower for phrase in abstention_phrases)
                )
                model_abstained_list.append(model_abstained)
            
            # Evaluate correctness
            correctness_scores = evaluator.evaluate_batch(
                llm_answers,
                reference_answers_list,
                should_abstain_list,
                model_abstained_list,
            )
            
            # Update samples with correctness labels
            num_evaluated = 0
            num_correct = 0
            num_incorrect = 0
            num_unknown = 0
            
            for sample, correctness, best_answer in zip(samples, correctness_scores, llm_answers):
                sample.correctness = correctness
                sample.answer = best_answer  # Store the best answer
                
                if correctness is not None:
                    num_evaluated += 1
                    if correctness >= 0.5:
                        num_correct += 1
                    else:
                        num_incorrect += 1
                else:
                    num_unknown += 1
            
            logger.info(
                f"  {split_name}: {num_evaluated} evaluated "
                f"({num_correct} correct, {num_incorrect} incorrect, {num_unknown} unknown)"
            )
        
        # Log overall statistics
        all_samples = data_split.train + data_split.calibration + data_split.test
        all_correctness = [s.correctness for s in all_samples if s.correctness is not None]
        
        if all_correctness:
            logger.info(f"\nOverall correctness statistics:")
            logger.info(f"  Total with labels: {len(all_correctness)}")
            logger.info(f"  Correct: {sum(1 for c in all_correctness if c >= 0.5)}")
            logger.info(f"  Incorrect: {sum(1 for c in all_correctness if c < 0.5)}")
            logger.info(f"  Mean correctness: {np.mean(all_correctness):.3f}")
        else:
            logger.error("No correctness labels evaluated! This will cause pipeline to fail.")
            raise ValueError(
                "No correctness labels could be evaluated. "
                "Check that samples have reference_answers or should_abstain in metadata."
            )
        
        # Check for edge case: all labels are the same
        if len(set(all_correctness)) == 1:
            logger.warning(
                f"⚠ All correctness labels are the same ({all_correctness[0]})! "
                "This may indicate an issue with correctness evaluation."
            )
        
        # Check for edge case: very few correct/incorrect samples
        correct_count = sum(1 for c in all_correctness if c >= 0.5)
        incorrect_count = sum(1 for c in all_correctness if c < 0.5)
        if correct_count < 5:
            logger.warning(f"⚠ Very few correct samples ({correct_count}). Calibration may be unreliable.")
        if incorrect_count < 5:
            logger.warning(f"⚠ Very few incorrect samples ({incorrect_count}). CRC threshold may be unreliable.")
        
        return data_split
    
    def _extract_features(self, generations, data_split):
        """Extract evidence features."""
        extractor = create_evidence_extractor()
        
        features = {}
        for split_type in [SplitType.TRAIN, SplitType.CALIBRATION, SplitType.TEST]:
            split_name = split_type.value
            samples = data_split.get_split(split_type)
            sample_ids = [s.sample_id for s in samples]
            
            logger.info(f"Extracting features for {split_name}...")
            feats = extractor.extract_batch(generations[split_name], sample_ids)
            features[split_name] = feats
        
        return features
    
    def _compute_uncertainties(self, generations):
        """Compute uncertainties."""
        estimator = create_uncertainty_estimator(
            methods=[self.config.uncertainty_method]
        )
        
        uncertainties = {}
        for split_name, gens in generations.items():
            logger.info(f"Computing uncertainties for {split_name}...")
            
            generations_list = [g.generations for g in gens]
            token_logprobs_list = [g.token_logprobs for g in gens]
            
            uncerts = estimator.compute_batch(
                generations_list,
                token_logprobs_list,
                method=self.config.uncertainty_method
            )
            
            uncertainties[split_name] = np.array(uncerts)
        
        return uncertainties
    
    def _prepare_weight_features(self, features):
        """Prepare features for NE-CRC weights."""
        # Use evidence features as weight features (could use embeddings instead)
        weight_features = {}
        
        for split_name, feats in features.items():
            # Convert to array
            feat_arrays = np.array([f.to_array() for f in feats])
            weight_features[split_name] = feat_arrays
        
        return weight_features
    
    def _run_all_systems(self, features, uncertainties, weight_features, data_split):
        """Run all system variants."""
        results = {}
        
        # Filter out samples with None correctness labels
        def filter_valid_samples(samples, feat_list, unc_list=None, wf_list=None):
            """Filter samples and corresponding features/uncertainties that have valid labels."""
            valid_indices = [i for i, s in enumerate(samples) if s.correctness is not None]
            filtered_samples = [samples[i] for i in valid_indices]
            filtered_features = [feat_list[i] for i in valid_indices]
            filtered_unc = [unc_list[i] for i in valid_indices] if unc_list is not None else None
            filtered_wf = [wf_list[i] for i in valid_indices] if wf_list is not None else None
            return filtered_samples, filtered_features, filtered_unc, filtered_wf
        
        # Filter train, calibration, and test sets
        train_samples, train_features_filtered, _, _ = filter_valid_samples(
            data_split.train, features['train']
        )
        cal_samples, cal_features_filtered, cal_unc_filtered, cal_wf_filtered = filter_valid_samples(
            data_split.calibration, features['calibration'], 
            uncertainties['calibration'], weight_features['calibration']
        )
        test_samples, test_features_filtered, test_unc_filtered, test_wf_filtered = filter_valid_samples(
            data_split.test, features['test'],
            uncertainties['test'], weight_features['test']
        )
        
        # Extract labels (now guaranteed to be non-None)
        train_labels = [s.correctness for s in train_samples]
        cal_labels = [s.correctness for s in cal_samples]
        test_labels = [s.correctness for s in test_samples]
        
        logger.info(f"\nFiltered samples with valid correctness labels:")
        logger.info(f"  Train: {len(train_samples)}/{len(data_split.train)} ({len(train_samples)/len(data_split.train)*100:.1f}%)")
        logger.info(f"  Calibration: {len(cal_samples)}/{len(data_split.calibration)} ({len(cal_samples)/len(data_split.calibration)*100:.1f}%)")
        logger.info(f"  Test: {len(test_samples)}/{len(data_split.test)} ({len(test_samples)/len(data_split.test)*100:.1f}%)")
        
        # Validate we have enough samples
        if len(train_samples) < 10:
            raise ValueError(f"Too few train samples with valid labels: {len(train_samples)}")
        if len(cal_samples) < 10:
            raise ValueError(f"Too few calibration samples with valid labels: {len(cal_samples)}")
        if len(test_samples) < 5:
            raise ValueError(f"Too few test samples with valid labels: {len(test_samples)}")
        
        # Log label distribution
        def log_label_distribution(labels, split_name):
            labels_arr = np.array(labels)
            correct = np.sum(labels_arr >= 0.5)
            incorrect = np.sum(labels_arr < 0.5)
            logger.info(f"  {split_name} label distribution: {correct} correct, {incorrect} incorrect ({correct/len(labels)*100:.1f}% correct)")
        
        logger.info(f"\nLabel distributions:")
        log_label_distribution(train_labels, "Train")
        log_label_distribution(cal_labels, "Calibration")
        log_label_distribution(test_labels, "Test")
        
        # Check for edge cases
        if len(set(cal_labels)) == 1:
            logger.warning(f"⚠ All calibration labels are the same ({cal_labels[0]})! This may cause issues with CRC.")
            # If all labels are 1.0, CRC will have no error samples to compute threshold
            if cal_labels[0] >= 0.5:
                logger.error("All calibration labels are correct (1.0). Cannot compute CRC threshold on error samples!")
                raise ValueError(
                    "Cannot compute CRC threshold: all calibration samples are correct. "
                    "Need at least some incorrect samples for risk control."
                )
        
        if len(set(train_labels)) == 1:
            logger.warning(f"⚠ All train labels are the same ({train_labels[0]})! Calibration head may not learn properly.")
        
        # Check calibration set has both correct and incorrect samples
        cal_correct = sum(1 for l in cal_labels if l >= 0.5)
        cal_incorrect = sum(1 for l in cal_labels if l < 0.5)
        if cal_incorrect == 0:
            logger.error("No incorrect samples in calibration set! Cannot compute CRC threshold.")
            raise ValueError(
                "Calibration set must contain at least some incorrect samples "
                "to compute CRC threshold for risk control."
            )
        if cal_correct == 0:
            logger.warning("No correct samples in calibration set! This is unusual.")
        
        logger.info(f"  Calibration set: {cal_correct} correct, {cal_incorrect} incorrect")
        
        # 1. Heuristic baseline
        logger.info("Running Heuristic baseline...")
        heuristic = create_heuristic_baseline(threshold=1.0, invert=True)
        results['heuristic'] = heuristic.predict(test_unc_filtered)
        
        # 2. UniCR (standard CRC)
        logger.info("Running UniCR baseline...")
        unicr = create_unicr_baseline(alpha=self.config.alpha)
        unicr.fit(
            train_features_filtered, train_labels,
            cal_features_filtered, cal_labels
        )
        
        # Log calibration head performance
        train_probs = unicr.calibration_head.predict_proba(train_features_filtered)
        train_preds = unicr.calibration_head.predict(train_features_filtered)
        train_acc = np.mean(train_preds == np.array(train_labels))
        logger.info(f"  Calibration head train accuracy: {train_acc:.3f}")
        logger.info(f"  Calibration head confidence range: [{train_probs.min():.3f}, {train_probs.max():.3f}]")
        
        # Log CRC threshold
        threshold = unicr.crc.get_threshold()
        logger.info(f"  CRC threshold: {threshold:.4f} (alpha={self.config.alpha})")
        
        results['unicr'] = unicr.predict(test_features_filtered)
        
        # 3. UniCR + Filter
        logger.info("Running UniCR + Filter...")
        unicr_filter = create_unicr_with_filter(
            alpha=self.config.alpha,
            delta=self.config.delta
        )
        unicr_filter.fit(
            train_features_filtered, train_labels,
            cal_features_filtered, cal_labels,
            cal_unc_filtered
        )
        
        # Log filter statistics
        test_pvalues = []
        for unc in test_unc_filtered:
            pval = unicr_filter.filter.compute_pvalue(unc)
            test_pvalues.append(pval)
        outlier_rate = sum(1 for p in test_pvalues if p < self.config.delta) / len(test_pvalues)
        logger.info(f"  SConU filter: {outlier_rate*100:.1f}% outliers detected (delta={self.config.delta})")
        
        results['unicr_filter'] = unicr_filter.predict(
            test_features_filtered, test_unc_filtered
        )
        
        # 4. UniCR + NE-CRC
        logger.info("Running UniCR + NE-CRC...")
        weight_fn = create_similarity_weights()
        unicr_necrc = create_unicr_with_necrc(
            alpha=self.config.alpha,
            weight_fn=weight_fn
        )
        unicr_necrc.fit(
            train_features_filtered, train_labels,
            cal_features_filtered, cal_labels,
            cal_wf_filtered
        )
        results['unicr_necrc'] = unicr_necrc.predict(
            test_features_filtered, test_wf_filtered
        )
        
        # 5. S-UniCR (full system)
        logger.info("Running S-UniCR (proposed)...")
        s_unicr = create_s_unicr(
            alpha=self.config.alpha,
            delta=self.config.delta,
            weight_fn=weight_fn
        )
        s_unicr.fit(
            train_features_filtered, train_labels,
            cal_features_filtered, cal_labels,
            cal_unc_filtered,
            cal_wf_filtered
        )
        
        # Log S-UniCR statistics
        test_thresholds = []
        for i in range(len(test_features_filtered)):
            test_feat = test_wf_filtered[i]
            threshold = s_unicr.ne_crc.compute_threshold(test_feat)
            test_thresholds.append(threshold)
        logger.info(f"  NE-CRC adaptive thresholds: mean={np.mean(test_thresholds):.4f}, std={np.std(test_thresholds):.4f}")
        
        results['s_unicr'] = s_unicr.predict(
            test_features_filtered,
            test_unc_filtered,
            test_wf_filtered
        )
        
        # Return results along with filtered test labels for evaluation
        return results, test_labels
    
    def _evaluate_metrics(self, results, test_labels):
        """Evaluate all metrics.
        
        Args:
            results: Dictionary of system results
            test_labels: List of test correctness labels (already filtered)
        """
        metrics_computer = create_metrics_computer()
        
        test_labels = np.array(test_labels)
        
        all_metrics = {}
        
        logger.info(f"\nEvaluating metrics on {len(test_labels)} test samples...")
        
        for system_name, preds in results.items():
            logger.info(f"\nEvaluating {system_name}...")
            
            # Extract arrays
            decisions = np.array([1 if r.is_answered() else 0 for r in preds])
            confidences = np.array([r.confidence for r in preds])
            
            # Compute metrics
            metrics = metrics_computer.compute_all(
                decisions,
                confidences,
                test_labels,
                alpha=self.config.alpha
            )
            
            all_metrics[system_name] = metrics
            
            # Log detailed summary
            answered = np.sum(decisions)
            logger.info(f"  Answered: {answered}/{len(decisions)} ({metrics['coverage']:.1%})")
            logger.info(f"  Selective risk: {metrics['selective_risk']:.3f} (target: {self.config.alpha:.3f})")
            logger.info(f"  Risk gap: {metrics.get('risk_gap', 0):.3f}")
            logger.info(f"  AURC: {metrics['aurc']:.4f}")
            logger.info(f"  Coverage@Risk≤5%: {metrics.get('coverage@risk<=0.05', 0):.3f}")
            
            # Check if risk constraint is satisfied
            if metrics.get('is_violated', False):
                logger.warning(f"  ⚠ Risk constraint VIOLATED (risk > target)")
            else:
                logger.info(f"  ✓ Risk constraint satisfied")
        
        return all_metrics
    
    def _save_results(self, results, metrics):
        """Save results to disk."""
        import pickle
        import json
        
        # Save raw results
        with open(self.output_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save metrics
        with open(self.output_dir / "metrics.pkl", 'wb') as f:
            pickle.dump(metrics, f)
        
        # Save metrics as JSON (excluding arrays)
        metrics_json = {}
        for system, system_metrics in metrics.items():
            metrics_json[system] = {
                k: v for k, v in system_metrics.items()
                if not k.startswith('_') and isinstance(v, (int, float, bool))
            }
        
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def create_experiment_pipeline(config: ExperimentConfig) -> ExperimentPipeline:
    """Factory function to create experiment pipeline.
    
    Args:
        config: Experiment configuration
    
    Returns:
        ExperimentPipeline instance
    """
    return ExperimentPipeline(config)

