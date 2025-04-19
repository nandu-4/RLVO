# Copyright (c) VCIP-NKU. All rights reserved.

import torch    
from typing import Optional, List, Dict

from .llm_wrapper import LLMWrapper

from peft import get_peft_model, LoraConfig
from real_lod.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from real_lod.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration


class ChatGLMWrapper(LLMWrapper):
    """
    A wrapper class for the ChatGLM model, extending the LLMWrapper class.

    This class provides methods to initialize and interact with the ChatGLM 
    model for text generation tasks, with optional support for parameter-efficient 
    fine-tuning (PEFT) using LoRA.

    Attributes:
        peft_config (Optional[Dict]): Configuration for parameter-efficient fine-tuning (PEFT).
        tokenizer (ChatGLMTokenizer): The tokenizer instance for the ChatGLM model.
        model (ChatGLMForConditionalGeneration): The ChatGLM model instance.
        backend (str): The backend framework used for the model (inherited from LLMWrapper).
        model_name (str): The name or path of the pre-trained model to load (inherited from LLMWrapper).
        device_map (str): The device configuration for model loading (e.g., "cpu", "cuda") (inherited from LLMWrapper).
        model_kwargs (dict): Additional arguments for loading the model (inherited from LLMWrapper).
        ckpt_path (Optional[str]): Path to a checkpoint file for loading model weights (inherited from LLMWrapper).
        generate_args (dict): Arguments for text generation, such as max_length, temperature, etc. (inherited from LLMWrapper).

    Methods:
        init_model():
            Initializes the tokenizer and model based on the specified backend and model name.
            Supports loading LoRA configurations for PEFT and loading model weights from a checkpoint.

        __call__(prompt: str, stop: Optional[List[str]] = None) -> str:
            Generates a response for a given prompt using the ChatGLM model.
    """
    def __init__(self, *args, 
                 peft_config: Optional[Dict] = None,
                 **kwargs):
        self.peft_config = peft_config
        super().__init__(*args, **kwargs)
        
    def init_model(self):
        if self.backend == "huggingface":
            self.tokenizer = ChatGLMTokenizer.from_pretrained(
                self.model_name,
            )
            self.model = ChatGLMForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                **self.model_args
            )
            
            if self.peft_config is not None:
                if self.peft_config.pop("peft_type", "lora") == "lora":
                    peft_config = LoraConfig(**self.peft_config)
                    self.model = get_peft_model(self.model, peft_config)
                else:
                    raise ValueError(f"Unsupported peft type: {self.peft_config['peft_type']}")
            
            if self.ckpt_path is not None:
                warning = self.model.load_state_dict(torch.load(self.ckpt_path), strict=False)
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
        response, _ = self.model.chat(self.tokenizer, prompt, **self.generate_args)
        return response



