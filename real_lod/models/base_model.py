# Copyright (c) VCIP-NKU. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class BaseModelWrapper(ABC):
    """
    An abstract base class for machine learning models.

    This class defines the common interface and attributes for all model implementations. 
    Subclasses must implement the `init_model` and `__call__` methods to define 
    model initialization and inference behavior.

    Attributes:
        model_name (str): The name or path of the pre-trained model to load.
        backend (str): The backend framework used for the model (e.g., "huggingface", "tensorflow").
        ckpt_path (Optional[str]): Path to a checkpoint file for loading model weights.
        device_map (Optional[str]): The device configuration for model loading (e.g., "cpu", "cuda").
        model_args (Optional[Dict[str, Any]]): Additional arguments for loading the model.
        generate_args (Optional[Dict[str, Any]]): Arguments for text generation, such as max_length, temperature, etc.

    Methods:
        init_model():
            Abstract method to initialize or load the model architecture and weights.
            Must be implemented by subclasses.

        __call__(*args, **kwargs) -> Any:
            Abstract method to run inference or forward pass of the model.
            Must be implemented by subclasses.
    """
    def __init__(self, 
                 model_name: str,
                 backend: str,
                 ckpt_path: Optional[str] = None,
                 device_map: Optional[str] = None,
                 model_args: Optional[Dict[str, Any]] = None,
                 generate_args: Optional[Dict[str, Any]] = None) -> None:
        self.model_name = model_name
        self.backend = backend
        self.ckpt_path = ckpt_path
        self.device_map = device_map
        self.model_args = model_args
        self.generate_args = generate_args
        
        self.init_model()
    
    @abstractmethod
    def init_model(self) -> None:
        """
        Initialize or load the model architecture and weights.

        This method must be implemented by any subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Run inference or forward pass of the model.

        Returns:
            Any: Model output.
        """
        raise NotImplementedError