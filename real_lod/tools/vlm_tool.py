# Copyright (c) VCIP-NKU. All rights reserved.

import requests
from io import BytesIO
from PIL import Image
from typing import Type, Optional, List, Any, Dict

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
from real_lod.models.base_model import BaseModelWrapper


class VLMToolPayload(BaseModel):
    prompt: Any = Field(..., description="The prompt about the image")
    input_info: Dict = Field(..., description="The input information of the image")
    
class VLMTool(BaseTool):
    """
    A tool for interacting with a vision-language model (VLM) as part of a workflow.

    This class extends the BaseTool class and provides methods for processing images, 
    applying image editing operations, and generating responses using a VLM backend.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        args_schema (Type[BaseModel]): The schema for the tool's input arguments.
        model (BaseModelWrapper): The VLM backend used for generating responses.
        template_name (Optional[str]): The name of the conversation template to use.
        input_variables (List): A list of input variables required by the tool.

    Methods:
        object_crop(image, bbox, scale_factor=2) -> Image:
            Crops the image around the specified bounding box with an optional scale factor.

        highlight(img, bbox, color=(255, 0, 0), thickness_scale=0.01) -> Image:
            Highlights the specified bounding box on the image with a colored rectangle.

        _load_image(path_or_url: str) -> Image:
            Loads an image from a file path or URL.

        _run(input_info: Dict, prompt: Any) -> str:
            Processes the input image and prompt, applies image editing, and generates a response.

        _arun(*args, **kwargs):
            Raises a NotImplementedError as asynchronous execution is not supported.
    """
    name: str = "vlm_tool"
    description: str = "Use VLM model as a tool to reperception image"
    args_schema: Type[BaseModel] = VLMToolPayload
    model: BaseModelWrapper = Field(..., description="The VLM model for tool use")
    template_name: Optional[str] = None
    input_variables: List[str] = ["input_info", "prompt"]

    def object_crop(self, image, bbox, scale_factor=2):
        w, h = image.width, image.height
        x_min, y_min, box_width, box_height = bbox
        x_max = x_min + box_width
        y_max = y_min + box_height

        # Calculate the expanded boundaries
        x_min = max(0, int(x_min - (scale_factor - 1) * box_width / 2))
        y_min = max(0, int(y_min - (scale_factor - 1) * box_height / 2))
        x_max = min(w, int(x_max + (scale_factor - 1) * box_width / 2))
        y_max = min(h, int(y_max + (scale_factor - 1) * box_height / 2))

        return image.crop((x_min, y_min, x_max, y_max))

    def highlight(self, img, bbox, color=(255, 0, 0), thickness_scale=0.01):
        
        x_min, y_min, box_width, box_height = bbox
        x_max = x_min + box_width
        y_max = y_min + box_height
        
        line_thickness = max(1, int(min(img.width, img.height) * thickness_scale))

        draw_img = ImageDraw.Draw(img)
        draw_img.rectangle((x_min, y_min, x_max, y_max), outline=color, width=line_thickness)
        return img
    
    def _load_image(self, path_or_url: str) -> Image.Image:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(path_or_url).convert("RGB")

    def _run(self, 
             input_info: Dict,
             prompt: Any) -> str:
        
        prompt_string = ""
        for prompt_item in prompt:
            prompt_string += f"{prompt_item.content}"
        
        image_path_or_url = input_info.get("image_path_or_url")
        
        image_editing = input_info.get("image_editing")
        bbox = input_info.get("bbox")
        
        image = self._load_image(image_path_or_url)
        
        # Suggested fix for the logic block inside the _run method

        bbox = list(map(int, bbox))
        edit_mode = image_editing.lower()

        if "extended object crop" in edit_mode:
            # First, highlight the box on the original image, then crop with an extension
            image = self.highlight(image, bbox)
            image = self.object_crop(image, bbox, scale_factor=2) # Using a scale factor > 1
        elif "object crop" in edit_mode:
            # Just crop the object with no extension
            image = self.object_crop(image, bbox, scale_factor=1)
        elif "object highlight" in edit_mode:
            # Just highlight the object
            image = self.highlight(image, bbox)
        else:
            raise ValueError(f"Unsupported image editing: {image_editing}")
        
        
        output = self.model(
            prompt=prompt_string,
            image=image
        )
        return output

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported for LLaVATool")