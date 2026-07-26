# Copyright (c) VCIP-NKU. All rights reserved.

# This module provides utility functions for various tasks such as bounding box normalization,
# processing examples for prompt templates, estimating GPU memory requirements, and handling YAML files.

# Functions:
# - uni_bbox: Normalizes bounding box coordinates relative to image dimensions.
# - process_examples: Processes a list of examples into a format suitable for prompt templates.
# - build_prompt_template: Constructs a ChatPromptTemplate using a dictionary of prompt components.
# - estimate_required_memory_mb: Estimates GPU memory usage based on the model name.
# - load_yaml_recursive: Recursively loads and merges YAML files, handling nested references.
# - args_to_dict: Converts command-line arguments into dictionaries, loading YAML files recursively if needed.

import os
import yaml

from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate

def uni_bbox(bbox, width, height):
    """
    Normalizes bounding box coordinates relative to image dimensions.

    Args:
        bbox (tuple): A tuple containing bounding box coordinates (x, y, w, h).
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        list: A list of normalized bounding box coordinates [x, y, w, h].
    """
    x, y, w, h = bbox
    return [round(x / width, 2), round(y / height, 2), round(w / width, 2), round(h / height, 2)]

def process_examples(examples: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Processes a list of examples into a format suitable for prompt templates.

    Args:
        examples (List[Dict[str, Any]]): A list of dictionaries containing "user" and "assistant" keys.

    Returns:
        List[Tuple[str, str]]: A list of tuples representing processed examples.
    """
    processed_examples = []
    for example in examples:
        processed_example = [
            ("user", example.get("user", "")),
            ("assistant", example.get("assistant", "")),
        ]
        processed_examples.extend(processed_example)
    return processed_examples

def build_prompt_template(prompt_template: Dict) -> ChatPromptTemplate:
    """
    Constructs a ChatPromptTemplate using a dictionary of prompt components.

    Args:
        prompt_template (Dict): A dictionary containing "system", "examples", and "user" keys.

    Returns:
        ChatPromptTemplate: A ChatPromptTemplate object constructed from the input dictionary.
    """
    messages = []
    if "system" in prompt_template:
        messages.append(("system", prompt_template["system"]))
        
    messages += process_examples(prompt_template.get("examples", []))
    if "user" in prompt_template:
        messages.append(("user", prompt_template["user"]))
    
    if "assistant" in prompt_template:
        messages.append(("assistant", prompt_template["assistant"]))
    
    prompt_template = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    return prompt_template

def estimate_required_memory_mb(model_name: str) -> int:
    """Estimated GPU memory usage in MB"""
    name = model_name.lower()
    if "34b" in name:
        return 6.5 * 1e4
    elif "13b" in name:
        return 2.5 * 1e4
    elif "7b" in name or "6b" in name:
        return 1.3 * 1e4
    else:
        return 1e4

def load_yaml_recursive(file_path, loaded_files=None):
    """
    Recursively loads a YAML file and merges any nested YAML references directly into the parent dictionary.
    Args:
        file_path (str): The path to the YAML file.
        loaded_files (set): A set of already loaded file paths to avoid circular dependencies.
    Returns:
        dict: The flattened dictionary from the YAML file and its nested references.
    """
    if loaded_files is None:
        loaded_files = set()

    normalized_path = os.path.abspath(file_path)

    if normalized_path in loaded_files:
        raise ValueError(f"Circular reference detected for file: {file_path}")

    loaded_files.add(normalized_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    flat_data = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and value.endswith('.yaml'):
                # nested_path = os.path.join(os.path.dirname(file_path), value)
                nested_path = value
                nested_data = load_yaml_recursive(nested_path, loaded_files)
                if key not in flat_data:
                    flat_data[key] = nested_data
                else:
                    flat_data[key].update(nested_data)  # Merge nested YAML data into parent
            else:
                flat_data[key] = value

    return flat_data

def args_to_dict(args):
    """
    Converts the parsed arguments into a single dictionary by loading each YAML file recursively.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        dict: A dictionary representation of the arguments.
        dict: A dictionary representation of YAML configurations.
    """
    args_dict = vars(args)  # Convert Namespace to a dictionary
    result_dict = {}
    result_dict_yaml = {}

    for key, value in args_dict.items():
        if isinstance(value, str) and value.endswith('.yaml'):
            result_dict_yaml[key] = load_yaml_recursive(value)  # Load YAML files recursively
        else:
            result_dict[key] = value

    return result_dict, result_dict_yaml