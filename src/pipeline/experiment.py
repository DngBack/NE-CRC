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
        logger.info("\n[2/8] Running LLM inference...")
        generations = self._run_llm_inference(data_split)
        
        # 3. Extract evidence features
        logger.info("\n[3/8] Extracting evidence features...")
        features = self._extract_features(generations, data_split)
        
        # 4. Compute uncertainties
        logger.info("\n[4/8] Computing uncertainties...")
        uncertainties = self._compute_uncertainties(generations)
        
        # 5. Prepare weight features for NE-CRC
        logger.info("\n[5/8] Preparing weight features...")
        weight_features = self._prepare_weight_features(features)
        
        # 6. Run all systems
        logger.info("\n[6/8] Running all system variants...")
        results, test_labels_filtered = self._run_all_systems(
            features, uncertainties, weight_features, data_split
        )
        
        # 7. Evaluate metrics
        logger.info("\n[7/8] Evaluating metrics...")
        metrics = self._evaluate_metrics(results, test_labels_filtered)
        
        # 8. Save results
        logger.info("\n[8/8] Saving results...")
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
        
        # Create split
        shift_type = self.config.shift_type
        data_split = self.splitter.create_split(
            samples,
            shift_type,
            self.config.split_ratios,
            dataset_name
        )
        
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Shift type: {shift_type.value}")
        logger.info(f"Train: {len(data_split.train)} samples")
        logger.info(f"Calibration: {len(data_split.calibration)} samples")
        logger.info(f"Test: {len(data_split.test)} samples")
        
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
        
        logger.info(f"Using {len(train_samples)}/{len(data_split.train)} train samples with valid labels")
        logger.info(f"Using {len(cal_samples)}/{len(data_split.calibration)} calibration samples with valid labels")
        logger.info(f"Using {len(test_samples)}/{len(data_split.test)} test samples with valid labels")
        
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
        
        for system_name, preds in results.items():
            logger.info(f"Evaluating {system_name}...")
            
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
            
            # Log summary
            logger.info(f"  Coverage: {metrics['coverage']:.3f}")
            logger.info(f"  Selective risk: {metrics['selective_risk']:.3f}")
            logger.info(f"  AURC: {metrics['aurc']:.4f}")
        
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

