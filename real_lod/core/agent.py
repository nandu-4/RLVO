# Copyright (c) VCIP-NKU. All rights reserved.

from pydantic import PrivateAttr
from typing import Optional, List

from langchain_core.outputs import LLMResult
from langchain_core.language_models.llms import LLM
from langchain_core.prompt_values import ChatPromptValue

from real_lod.models.base_model import BaseModelWrapper
from real_lod.utils.conversation import get_conv_template, Conversation


ROLE_INDEX = {
    "human": 0,
    "ai": 1
}

class Agent(LLM):
    """
    A class representing an agent for handling language model interactions.

    This class extends the LLM class and provides methods for generating prompts,
    managing conversations, and interacting with a language model backend.

    Attributes:
        _model (BaseModelWrapper): The language model backend used for generating responses.
        template_name (Optional[str]): The name of the conversation template to use.
        conversation (Optional[Conversation]): The conversation object for managing dialogue history.

    Methods:
        generate_prompt(prompts, stop=None, callbacks=None, **kwargs) -> List[str]:
            Processes input prompts, appends them to the conversation, and generates prompt strings.

        _call(prompt: str, stop: Optional[List[str]] = None) -> str:
            Generates a response for a single prompt using the model.

        _generate(prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
            Generates responses for a list of prompts.

    Properties:
        _llm_type -> str:
            Returns the type of the language model backend.
    """
    _model: BaseModelWrapper = PrivateAttr(default=None)
    template_name: Optional[str] = None
    conversation: Optional[Conversation] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Manually assign the private attribute
        self._model = kwargs.get('model')
        self._init_conversation()

    def _init_conversation(self):
        self.conversation = get_conv_template(self.template_name)

    @property
    def _llm_type(self) -> str:
        return f"{self._model.backend}_{self._model.model_name}_llm"

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """
        Generates prompts by processing input prompts and appending them to the conversation.

        Args:
            prompts (List[Union[ChatPromptValue, str]]): A list of prompts, either as ChatPromptValue objects or strings.
            stop (Optional[List[str]]): A list of stop tokens to control the generation process.
            callbacks (Optional[Any]): Callbacks for handling intermediate results or events during generation.
            **kwargs: Additional keyword arguments for the generation process.

        Returns:
            List[str]: A list of generated prompt strings.
        """
        self._init_conversation()

        prompt_strings = []
        for prompt in prompts:
            if isinstance(prompt, ChatPromptValue):
                messages = prompt.to_json()["kwargs"]["messages"]
                for message in messages:
                    if message.type == "system":
                        self.conversation.set_system_message(message.content)
                    else:
                        role_index = ROLE_INDEX.get(message.type)
                        if role_index is None:
                            raise ValueError(f"Invalid role: {message.type}")
                        self.conversation.append_message(self.conversation.roles[role_index], message.content)
            else:
                prompt = prompt.to_string()
                self.conversation.append_message(self.conversation.roles[0], prompt)

            self.conversation.append_message(self.conversation.roles[1], None)
            prompt_strings.append(self.conversation.get_prompt())

        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response for a single prompt using the model.

        Args:
            prompt (str): The input prompt string.
            stop (Optional[List[str]]): A list of stop tokens to control the generation process.

        Returns:
            str: The generated response string.
        """

        response = self._model(prompt)
        return response

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """
        Generates responses for a list of prompts.

        Args:
            prompts (List[str]): A list of input prompt strings.
            stop (Optional[List[str]]): A list of stop tokens to control the generation process.

        Returns:
            LLMResult: An object containing the generated responses.
        """
        generations = [[{"text": self._call(prompt)}] for prompt in prompts]
        return LLMResult(generations=generations)