# Copyright (c) VCIP-NKU. All rights reserved.

from typing import Type, Optional, Any, List

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from real_lod.models.base_model import BaseModelWrapper
from langchain_core.prompt_values import ChatPromptValue
from real_lod.utils.conversation import get_conv_template, Conversation

ROLE_INDEX = {
    "human": 0,
    "ai": 1
}

class LLMToolPayload(BaseModel):
    prompt: Any = Field(...)


class LLMTool(BaseTool):
    """
    A tool for interacting with a large language model (LLM) as part of a workflow.

    This class extends the BaseTool class and provides methods for processing prompts,
    managing conversations, and generating responses using an LLM backend.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        args_schema (Type[BaseModel]): The schema for the tool's input arguments.
        model (BaseModelWrapper): The LLM backend used for generating responses.
        template_name (Optional[str]): The name of the conversation template to use.
        conversation (Optional[Conversation]): The conversation object for managing dialogue history.
        input_variables (List): A list of input variables required by the tool.

    Methods:
        _init_conversation():
            Initializes the conversation object using the specified template.

        _run(prompt: ChatPromptValue) -> str:
            Processes the input prompt, appends it to the conversation, and generates a response.

        _arun(*args, **kwargs):
            Raises a NotImplementedError as asynchronous execution is not supported.
    """
    name: str = "llm_tool"
    description: str = "Use LLM model as a tool"
    args_schema: Type[BaseModel] = LLMToolPayload
    model: BaseModelWrapper = Field(...)
    template_name: Optional[str] = None
    conversation: Optional[Conversation] = None
    input_variables: List[str] = ["prompt"]

    def _init_conversation(self):
        self.conversation = get_conv_template(self.template_name)

    def _run(self, prompt: ChatPromptValue) -> str:
        self._init_conversation()
        for message in prompt:
            if message.type == "system":
                self.conversation.set_system_message(message.content)
            else:
                role_index = ROLE_INDEX.get(message.type)
                if role_index is None:
                    raise ValueError(f"Invalid role: {message.type}")
                self.conversation.append_message(self.conversation.roles[role_index], message.content)
        self.conversation.append_message(self.conversation.roles[1], None)


        output = self.model(self.conversation.get_prompt(), stop=None)
        output = output.split(self.conversation.roles[1] + ":")[-1].strip().replace("\n", "")
        return output


    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported for LLaVATool")