# Copyright (c) VCIP-NKU. All rights reserved.

import torch

from typing import Dict, List, Any

import re
import json
import copy
import random
import argparse
from mmengine.logging import MMLogger

import json_repair
from langchain.schema.runnable import RunnableMap

from real_lod.models import build_model_wrapper
from real_lod.tools.vlm_tool import VLMTool
from real_lod.tools.llm_tool import LLMTool
from real_lod.core.agent import Agent
from real_lod.utils.common_utils import args_to_dict, estimate_required_memory_mb, build_prompt_template, uni_bbox

import warnings
warnings.filterwarnings("ignore")

RESPONSE_PLANNING_PATTERN = r'"value"\s*:\s*"([^"]*)"'
REASONING_PLANNING_PATTERN = r'"thoughts"\s*:\s*"([^"]*)"'
ACTIONS_PLANNING_PATTERN = r'"actions"\s*:\s*"([^"]*)"'
REWRITE_PATTERN = r"New Expressions:\s*(.+)"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse various command-line arguments.")
    parser.add_argument("--configs", type=str, default="configs/real_lod.yaml", help="Path to the configuration file for agent.")
    parser.add_argument("--max_cycles", type=int, default=3, help="Maximum number of cycles for the workflow.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


class RealLOD:
    """
    A class representing the Real-LOD workflow for vision-language tasks.

    This class orchestrates the interaction between various models and tools, including 
    large language models (LLMs) and vision-language models (VLMs), to process and refine 
    input data through multiple stages such as planning, tool usage, reflection, and rewriting.

    Attributes:
        logger (MMLogger): Logger instance for logging workflow activities.
        config_agent (dict): Configuration for the agent model.
        config_rewriter (dict): Configuration for the rewriter model.
        config_reflector (dict): Configuration for the reflector model.
        config_vlm_tool (dict): Configuration for the vision-language model (VLM) tool.
        max_cycles (int): Maximum number of cycles for the workflow.
        model_dict (dict): Dictionary to store initialized models.
        agent_model (BaseModelWrapper): The agent model instance.
        rewriter_model (BaseModelWrapper): The rewriter model instance.
        reflector_model (BaseModelWrapper): The reflector model instance.
        vlm_tool_model (BaseModelWrapper): The vision-language model instance.
        agent (Agent): The agent instance for planning.
        rewriter (RunnableMap): The rewriter tool instance.
        reflector (RunnableMap): The reflector tool instance.
        vlm_tool (RunnableMap): The vision-language model tool instance.

    Methods:
        build_model(model_config: Dict) -> BaseModelWrapper:
            Initializes and returns a model based on the provided configuration.

        build_chain(config: Dict, model: Any = None, chain_cls: Any = None, is_tool: bool = False) -> Any:
            Builds and returns a processing chain for a model or tool.

        handle_device(model_name: str) -> torch.device:
            Determines the appropriate device (CPU/GPU) for model execution.

        init_info_pool(sample_info: dict) -> dict:
            Initializes the information pool for a given sample.

        planning(info_pool: dict) -> Tuple[dict, List[dict]]:
            Performs the planning stage, generating reasoning, actions, and responses.

        tool_use(info_pool: Dict, action: Dict) -> Tuple[dict, str]:
            Executes a tool (LLM or VLM) based on the specified action.

        reflection(info_pool: Dict, action_outputs: List[str] = []) -> dict:
            Performs the reflection stage to refine the workflow's output.

        rewrite(info_pool: Dict) -> dict:
            Executes the rewriter tool to refine the current expression.

        test(sample_info: Dict):
            Runs a test workflow on a given sample.

        refine(sample_info: Dict) -> dict:
            Executes the full refinement workflow on a given sample.
    """

    def __init__(self, 
                 configs, 
                 max_cycles=3,
                 debug=False):
        self.logger = MMLogger('Real-LOD', logger_name='real-lod', log_level='DEBUG' if debug else 'INFO')
        
        self.config_agent = configs.get("agent", {})
        self.config_rewriter = configs.get("rewriter", {})
        self.config_reflector = configs.get("reflector", {})
        self.config_vlm_tool = configs.get("vlm_tool", {})
        self.max_cycles = max_cycles
                
        # Initialize models
        self.model_dict = dict()
        # Initialize the agent model
        self.agent_model = self.build_model(self.config_agent["model_config"])
        # Initialize the rewiter model
        self.rewriter_model = self.build_model(self.config_rewriter["model_config"])
        # Initialize the reflector model
        self.reflector_model = self.build_model(self.config_reflector["model_config"])
        # Initialize the VLM model
        self.vlm_tool_model = self.build_model(self.config_vlm_tool["model_config"])
        
        
        # Initialize the agent
        self.logger.info(f"Initializing Agent")
        self.agent = self.build_chain(config=self.config_agent, 
                                           chain_cls=Agent, 
                                           model=self.agent_model)
        # Initialize the tool
        # Initialize the rewriter tool
        self.logger.info(f"Initializing Rewriter")
        self.rewriter = self.build_chain(config=self.config_rewriter,
                                         chain_cls=LLMTool,
                                         model=self.rewriter_model,
                                         is_tool=True)
        # Initialize the reflector tool
        self.logger.info(f"Initializing reflector")
        self.reflector = self.build_chain(config=self.config_reflector,
                                         chain_cls=LLMTool,
                                         model=self.reflector_model,
                                         is_tool=True)
        
        # Initialize the VLM tool
        self.logger.info(f"Initializing vlm tool")
        self.vlm_tool = self.build_chain(config=self.config_vlm_tool,
                                         chain_cls=VLMTool,
                                         model=self.vlm_tool_model,
                                         is_tool=True)
    
    def build_model(self,
                    model_config: Dict):
        model_name = model_config["model_name"]
        self.logger.info(f"Initializing model: {model_name}")

        if model_name in self.model_dict:
            self.logger.info(f"Model {model_name} already initialized, reusing the existing model.")
            return self.model_dict[model_name]
        
        device_map = self.handle_device(model_name)
        wrapper_type = model_config.pop("wrapper_type", None)
        wrapper_cls =  build_model_wrapper(wrapper_type)
        model = wrapper_cls(device_map=device_map, **model_config)
        self.model_dict[model_name] = model
        self.logger.info(f"Model {model_name} initialized successfully.")
        return model    
    
    def build_chain(self, 
                    config: Dict, 
                    model: Any = None,
                    chain_cls: Any = None,
                    is_tool: bool = False):
        template = config.pop("prompt_template", "")
        template = build_prompt_template(template)
        chain = chain_cls(template_name=config.get("template_name"), model=model)
        
        if is_tool:
            runnable_dict = {"prompt": lambda input: template.format_messages(**input)}
            for meta_key in chain.input_variables:
                if meta_key != "prompt":
                    print(f"Adding meta key: {meta_key}")
                    runnable_dict[meta_key] = lambda input, key=meta_key: input[key]
            
            template_runmap = RunnableMap(runnable_dict)
            return template_runmap | chain
        return template | chain
            
    def handle_device(self, model_name:str):
        # Check for available GPUs
        if torch.cuda.is_available():
            gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            gpus = None
        required = estimate_required_memory_mb(model_name)
        if gpus is None:
            return torch.device('cpu')
        if hasattr(torch.cuda, 'mem_get_info'):
            free_memories = [torch.cuda.mem_get_info(gpu)[0] // (1024 * 1024) for gpu in gpus]
            if all(free_memory < required for free_memory in free_memories):
                return "auto"
            select = max(zip(free_memories, range(len(free_memories))))[1]
        else:
            select = random.randint(0, len(gpus) - 1)
        return gpus[select]
    
    def init_info_pool(self, sample_info: dict):
        info_pool = dict(
            feedback="",
            is_solved=False,
            cycle_iter=1,
            history_messages=[]
        )
        
        info_pool["current_expression"] = sample_info["raw_expression"]
        info_pool["raw_expression"] = sample_info["raw_expression"]
        info_pool["category"] = sample_info["object_locations"]["chosen_object"]["category"]
        
        info_pool["image_path"] = sample_info["image_path"]
        info_pool["height"] = sample_info["height"]
        info_pool["width"] = sample_info["width"]

        info_pool["origin_chosen_object"] = copy.deepcopy(sample_info["object_locations"]["chosen_object"])
        sample_info["object_locations"]["chosen_object"]["bbox"] = uni_bbox(sample_info["object_locations"]["chosen_object"]["bbox"],  sample_info["width"],  sample_info["height"])
        info_pool["chosen_object"] = sample_info["object_locations"]["chosen_object"]
        info_pool["other_objects"] = sample_info["object_locations"]["other_objects"]
        info_pool["caption"] = sample_info["global_caption"]
        
        return info_pool
    
    def planning(self, info_pool: dict):
        reasoning = ""
        actions = []
        response = ""
            
        if info_pool["cycle_iter"] == 1:
            output = ""
            reasoning, actions, response = ("No LLM Reflection", [], "Need Reflection")
        else:
            current_expression = info_pool["current_expression"]
            category = info_pool["category"]
            feedback = info_pool["feedback"]
            self.logger.debug(f"Input of Planning: Expression: {current_expression} Category: {category} Feedback: {feedback}")
            
            output = self.agent.invoke(info_pool).lower()
            output = json_repair.loads(output)
            self.logger.debug("Output of Planning: " + str(output))
            reasoning = output.get("reasoning", "")
            actions = output.get("actions", [])
            response = output.get("response", "")

        history_message = {
                "stage": "planning",
                "reasoning": reasoning,
                "actions": actions,
                "response": response,
                "output": output
        }
        info_pool["history_messages"].append(history_message)
        return info_pool, actions
    
    def tool_use(self, info_pool: Dict, action: Dict):
        if "llm" in action['tool_name'].lower():
            return self.rewrite(info_pool), ""
        elif "vlm" in action['tool_name'].lower():
            input_info = dict(image_path_or_url=info_pool["image_path"],
                              bbox=info_pool["origin_chosen_object"]["bbox"],
                              image_editing=action['tool_parameters']['image_editing'])
            action_output = self.vlm_tool.invoke(dict(
                input_info = input_info,
                question=action['tool_parameters']['question'],
                category=info_pool["category"],
                current_expression=info_pool["current_expression"],
            ))
            
            self.logger.debug("Output of VLM Tool: " + action_output)
            
            info_pool["history_messages"].append({
                "stage": "tool_use",
                "action": action,
                "output": action_output
            })
        elif action['tool_name'] == 'none':
            action_output = ""
            info_pool["is_solved"] = True
        else:
            raise NotImplementedError(f"Invalid tool_name: {action['tool_name']}.")
        return info_pool, action_output
    
    def reflection(self, 
                   info_pool: Dict, 
                   action_outputs: List[str] = []):
        if info_pool["cycle_iter"] > 1 and len(action_outputs) == 0:
            info_pool["is_solved"] = True
        info_pool["caption"] += "\n".join(action_outputs) if len(action_outputs) > 0 else ""
        output = self.reflector.invoke(info_pool)        
        self.logger.debug("Output of Reflection: " + output)
        info_pool["feedback"] = output
        info_pool["history_messages"].append({
            "stage": "reflection",
            "output": output,
            "feedback": output
        })
        return info_pool
    
    def rewrite(self, info_pool: Dict):
        output = self.rewriter.invoke(info_pool)
        match = re.search(REWRITE_PATTERN, output)
        refined_expression = match.group(1) if match else info_pool["current_expression"]
        self.logger.debug("Output of Rewrite: " + refined_expression)
        info_pool["current_expression"] = refined_expression
        info_pool["history_messages"].append({
            "stage": "tool_use",
            "action": "LLM-rewrite",
            "output": output,
            "refined_expression": refined_expression
        })
        return info_pool
    
    def test(self, sample_info: Dict):
        info_pool = self.init_info_pool(sample_info)
        info_pool["cycle_iter"] += 1
        
        # Test the planning
        self.planning(info_pool)
        
        # Test the reflection
        self.reflection(info_pool)
        
        # Test the tool use
        self.tool_use(info_pool, action={"tool_name": "LLM"})
        self.tool_use(info_pool, action={"tool_name": "VLM", "tool_parameters": {"question": "Test", 
                                                                             "image_editing": "the object itself"}})
        self.tool_use(info_pool, action={"tool_name": "VLM", "tool_parameters": {"question": "Test", 
                                                                             "image_editing": "the object and the surrounding areas"}})
        self.tool_use(info_pool, action={"tool_name": "VLM", "tool_parameters": {"question": "Test", 
                                                                             "image_editing": "the whole image"}})        
        self.logger.debug(info_pool["history_messages"])
    
    def refine(self, sample_info: Dict):
        info_pool = self.init_info_pool(sample_info)
        
        stop = False
                
        while not stop:
            info_pool, actions = self.planning(info_pool)
            
            action_outputs = []
            for action in actions:
                info_pool, action_output = self.tool_use(info_pool, action)
                action_outputs.append(action_output)            
                
            info_pool = self.reflection(info_pool, action_outputs)
            
            stop = info_pool["is_solved"] or (info_pool["cycle_iter"]==self.max_cycles)
            info_pool["cycle_iter"] += 1
            
        return info_pool
    

if __name__ == "__main__":
    args = parse_arguments()
    args, configs = args_to_dict(args)
    
    sample_infos = [dict(
        image_path="assets/demo_images/real_lod/obj365_val_000000569935.jpg",
        height=427,
        width=640,
        raw_expression="A seal in the water",
        object_locations = dict(
            chosen_object={'id':0, 'category': 'seal', 'bbox': [105.9069824, 105.5966186429, 329.731689472, 266.762084974]},
            other_objects=[],
        ),
        global_caption ="The seal in the red bounding box is a dark brown color, with a smooth, wet skin. It is located in the center of the image, surrounded by water. The seal appears to be resting or swimming, with its head slightly raised above the water's surface. It is not interacting with any other objects outside the red bounding box, but it is in close proximity to several birds, which are also in the water. The birds are various shades of gray and white, and they are scattered around the seal, some closer and others further away. The water around the seal and the birds is murky, suggesting that it might be shallow or that there is a lot of sediment or algae in the water. The seal's position in the center of the image, along with its size and color, make it the focal point of the image. The birds, while numerous, are smaller and less prominent, serving as a backdrop to the seal."
    )]
    
    real_lod = RealLOD(configs=configs["configs"], 
                       **args)
    
    for sample_info in sample_infos:
        real_lod.test(sample_info)
        updated_sample_info = real_lod.refine(sample_info)



