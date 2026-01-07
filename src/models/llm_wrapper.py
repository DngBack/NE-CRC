"""LLM inference wrapper supporting vLLM and transformers."""

import time
from typing import List, Optional, Dict, Any, Tuple
import torch
from abc import ABC, abstractmethod
from loguru import logger

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use transformers instead")

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from ..data import GeneratedOutput


class BaseLLMInference(ABC):
    """Base class for LLM inference."""
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprobs: bool = True,
    ) -> List[GeneratedOutput]:
        """Generate responses for prompts.
        
        Args:
            prompts: List of input prompts
            num_samples: Number of samples per prompt (K)
            max_tokens: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            return_logprobs: Whether to return token log probabilities
        
        Returns:
            List of GeneratedOutput objects
        """
        pass
    
    @abstractmethod
    def get_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> List[float]:
        """Get token log probabilities for a specific completion.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
        
        Returns:
            List of token log probabilities
        """
        pass


class VLLMInference(BaseLLMInference):
    """VLLM-based inference for fast batch generation."""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        """Initialize VLLM model.
        
        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model sequence length
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Install with: pip install vllm")
        
        logger.info(f"Loading model with vLLM: {model_name}")
        self.model_name = model_name
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompts: List[str],
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprobs: bool = True,
    ) -> List[GeneratedOutput]:
        """Generate using vLLM."""
        start_time = time.time()
        
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=5 if return_logprobs else None,
        )
        
        logger.info(f"Generating {num_samples} samples for {len(prompts)} prompts")
        outputs = self.llm.generate(prompts, sampling_params)
        
        latency_ms = (time.time() - start_time) * 1000
        avg_latency = latency_ms / len(prompts)
        
        results = []
        for prompt, output in zip(prompts, outputs):
            generations = [o.text for o in output.outputs]
            
            # Extract token logprobs if available
            token_logprobs = None
            if return_logprobs and output.outputs[0].logprobs:
                token_logprobs = []
                for sample_output in output.outputs:
                    if sample_output.logprobs:
                        sample_logprobs = [
                            list(token_logprob.values())[0].logprob
                            for token_logprob in sample_output.logprobs
                        ]
                        token_logprobs.append(sample_logprobs)
            
            # Count tokens (approximate)
            total_tokens = sum(len(g.split()) for g in generations)
            
            result = GeneratedOutput(
                query=prompt,
                generations=generations,
                token_logprobs=token_logprobs,
                generation_params={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "num_samples": num_samples,
                },
                total_tokens=total_tokens,
                latency_ms=avg_latency,
            )
            results.append(result)
        
        logger.info(f"Generation complete. Avg latency: {avg_latency:.2f}ms")
        return results
    
    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Get logprobs for specific completion (single sample with prompt)."""
        full_text = prompt + completion
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=len(full_text.split()),
            logprobs=5,
        )
        
        output = self.llm.generate([full_text], sampling_params)[0]
        if output.outputs[0].logprobs:
            return [
                list(token_logprob.values())[0].logprob
                for token_logprob in output.outputs[0].logprobs
            ]
        return []


class TransformersInference(BaseLLMInference):
    """Transformers-based inference (fallback)."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize transformers model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
        """
        logger.info(f"Loading model with transformers: {model_name}")
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {"device_map": "auto" if device == "cuda" else None}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()
        
        logger.info(f"Model loaded on {device}")
    
    def generate(
        self,
        prompts: List[str],
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprobs: bool = True,
    ) -> List[GeneratedOutput]:
        """Generate using transformers."""
        results = []
        
        for prompt in prompts:
            start_time = time.time()
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]
            
            # Generate multiple samples
            generations = []
            all_logprobs = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_scores=return_logprobs,
                        return_dict_in_generate=True,
                    )
                
                # Decode generation
                generated_ids = outputs.sequences[0][input_length:]
                generation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generations.append(generation)
                
                # Extract logprobs if requested
                if return_logprobs and hasattr(outputs, 'scores') and outputs.scores:
                    logprobs = []
                    for score in outputs.scores:
                        probs = torch.softmax(score[0], dim=-1)
                        token_id = generated_ids[len(logprobs)]
                        logprob = torch.log(probs[token_id]).item()
                        logprobs.append(logprob)
                    all_logprobs.append(logprobs)
            
            latency_ms = (time.time() - start_time) * 1000
            total_tokens = sum(len(g.split()) for g in generations)
            
            result = GeneratedOutput(
                query=prompt,
                generations=generations,
                token_logprobs=all_logprobs if all_logprobs else None,
                generation_params={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "num_samples": num_samples,
                },
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )
            results.append(result)
        
        return results
    
    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Get logprobs for specific completion."""
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get logprobs for each token
        logprobs = []
        for i in range(1, inputs.input_ids.shape[1]):
            token_id = inputs.input_ids[0, i]
            probs = torch.softmax(logits[0, i - 1], dim=-1)
            logprob = torch.log(probs[token_id]).item()
            logprobs.append(logprob)
        
        return logprobs


def create_llm_inference(
    model_name: str,
    use_vllm: bool = True,
    **kwargs,
) -> BaseLLMInference:
    """Factory function to create LLM inference object.
    
    Args:
        model_name: HuggingFace model name or path
        use_vllm: Whether to use vLLM (if available)
        **kwargs: Additional arguments for specific backend
    
    Returns:
        LLM inference object
    """
    if use_vllm and VLLM_AVAILABLE:
        return VLLMInference(model_name, **kwargs)
    else:
        if use_vllm:
            logger.warning("vLLM requested but not available, falling back to transformers")
        return TransformersInference(model_name, **kwargs)


if __name__ == "__main__":
    # Example usage
    model_name = "meta-llama/Llama-3.2-1B"  # Small model for testing
    
    # Create inference engine
    llm = create_llm_inference(model_name, use_vllm=False)
    
    # Generate samples
    prompts = ["What is the capital of France?"]
    outputs = llm.generate(prompts, num_samples=3, max_tokens=50, temperature=0.8)
    
    print(f"\nGenerated {len(outputs)} outputs")
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i}: {output.query}")
        for j, gen in enumerate(output.generations):
            print(f"  Sample {j}: {gen[:100]}...")

