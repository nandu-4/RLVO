# Copyright (c) VCIP-NKU. All rights reserved.

from PIL import Image
from .base_model import BaseModelWrapper
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class LLaVAVLMWrapper(BaseModelWrapper):
    """
    A wrapper class for the LLaVA vision-language model (VLM), extending the BaseModelWrapper class.

    This class provides methods to initialize and interact with the LLaVA model for 
    vision-language tasks, such as answering questions based on an input image and text.

    Attributes:
        processor (LlavaProcessor): The processor instance for handling image and text inputs.
        model (LlavaForConditionalGeneration): The LLaVA model instance.
        backend (str): The backend framework used for the model (inherited from BaseModelWrapper).
        model_name (str): The name or path of the pre-trained model to load (inherited from BaseModelWrapper).
        device_map (Optional[str]): The device configuration for model loading (e.g., "cpu", "cuda") (inherited from BaseModelWrapper).
        model_kwargs (Optional[dict]): Additional arguments for loading the model (inherited from BaseModelWrapper).
        generate_args (Optional[dict]): Arguments for text generation, such as max_length, temperature, etc. (inherited from BaseModelWrapper).

    Methods:
        init_model():
            Initializes the processor and model based on the specified backend and model name.
            Currently supports the Hugging Face Transformers backend.

        __call__(question: str, image: Image) -> str:
            Generates a response to a question based on the provided image.

            Args:
                question (str): The input question as a string.
                image (Image): The input image as a PIL Image object.

            Returns:
                str: The generated response string.
    """
    def init_model(self):
        if self.backend == "huggingface":
            self.processor = LlavaProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                resume_download=True,
                device_map=self.device_map,
                **self.model_args
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def __call__(self, prompt: str, image: Image):
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.model.device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, **self.generate_args)
        output = self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1]
        return output