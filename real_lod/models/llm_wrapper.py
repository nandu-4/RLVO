# Copyright (c) VCIP-NKU. All rights reserved.

from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

from .base_model import BaseModelWrapper

class LLMWrapper(BaseModelWrapper):
    """
    A wrapper class for large language models (LLMs), extending the BaseModelWrapper class.

    This class provides methods to initialize and interact with pre-trained 
    language models for text generation tasks using the Hugging Face Transformers library.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer instance for the model.
        model (AutoModelForCausalLM): The loaded Hugging Face model instance.
        backend (str): The backend framework used for the model (inherited from BaseModelWrapper).
        model_name (str): The name or path of the pre-trained model to load (inherited from BaseModelWrapper).
        device_map (Optional[str]): The device configuration for model loading (e.g., "cpu", "cuda") (inherited from BaseModelWrapper).
        model_args (Optional[dict]): Additional arguments for loading the model (inherited from BaseModelWrapper).
        generate_args (Optional[dict]): Arguments for text generation, such as max_length, temperature, etc. (inherited from BaseModelWrapper).

    Methods:
        init_model():
            Initializes the tokenizer and model based on the specified backend and model name.
            Currently supports the Hugging Face Transformers backend.

        __call__(prompt: str, stop: Optional[List[str]] = None) -> str:
            Generates a response for a given prompt using the model.
    """

    def init_model(self):
        if self.backend == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                **self.model_args
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response for a single prompt using the model.

        Args:
            prompt (str): The input prompt string.
            stop (Optional[List[str]]): A list of stop tokens to control the generation process.

        Returns:
            str: The generated response string.
        """
        
        pipeline = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        response = pipeline(prompt, **self.generate_args)[0]["generated_text"]
        response = response
        return response



