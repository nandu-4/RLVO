# Copyright (c) VCIP-NKU. All rights reserved.

import os
import json
import argparse
from real_lod.workflow import RealLOD
from real_lod.utils.common_utils import args_to_dict

def parse_arguments():
    """
    Parses command-line arguments for the RealLOD workflow.

    Returns:
        argparse.Namespace: Parsed arguments including paths, configurations, and options.
    """
    parser = argparse.ArgumentParser(description="Parse various command-line arguments.")
    parser.add_argument("annotation", type=str, help="Path to the input annotation file.")
    parser.add_argument("--configs", type=str, default="configs/real_lod.yaml", help="Path to the configuration file for agent.")
    parser.add_argument("--max_cycles", type=int, default=3, help="Maximum number of cycles for the workflow.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--save_dir", type=str, default="save", help="Directory to save results.")
    return parser.parse_args()


def main():
    """
    Main function to execute the RealLOD workflow.

    This function initializes the RealLOD workflow, loads the input annotation file, 
    processes each sample through the refinement workflow, and appends the refined 
    results to the output list.
    """
    args = parse_arguments()
    args, configs = args_to_dict(args)
    
    annotations = args.pop("annotation", "")
    save_dir = args.pop("save_dir", "")
    
    # Initialize the RealLOD workflow with the provided arguments
    real_lod = RealLOD(configs=configs["configs"], **args)
    
    
    # Load the annotation file
    with open(annotations, "r") as f:
        sample_infos = json.load(f)
    
    refined_sample_infos = []
    for sample_info in sample_infos:
        update_info = real_lod.refine(sample_info)
        if update_info["is_solved"]:
            sample_info["history_messages"] = update_info["history_messages"]
            sample_info["refined_expression"] = update_info["current_expression"]
            refined_sample_infos.append(sample_info)
    
    os.makedirs(save_dir, exist_ok=True)
    # Save the refined sample information to a JSON file
    output_path = os.path.join(save_dir, "refined_sample_infos.json")
    with open(output_path, "w") as f:
        json.dump(refined_sample_infos, f, indent=4)
        

if __name__ == "__main__":
    main()