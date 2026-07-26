# Copyright (c) VCIP-NKU. All rights reserved.

from .chatglm_llm_wrapper import ChatGLMWrapper
from .llm_wrapper import LLMWrapper
from .llava_vlm_wrapper import LLaVAVLMWrapper


WRAPPER_MAPPING = {
    "chatglm": ChatGLMWrapper,
    "llm": LLMWrapper,
    "llava": LLaVAVLMWrapper
}

def build_model_wrapper(wrapper_type: str):
    if wrapper_type not in WRAPPER_MAPPING:
        raise ValueError(f"Unsupported model wrapper type: {wrapper_type}")
    return WRAPPER_MAPPING[wrapper_type]