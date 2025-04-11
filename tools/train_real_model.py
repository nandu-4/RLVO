# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

from real_model import *

def parse_args():
    """
    Parse command-line arguments for training a detector.

    This function creates an argument parser and defines various command-line
    arguments related to the training configuration, work directory, automatic
    mixed precision, learning rate scaling, resume options, and distributed training.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description='Train a detector')
    # Add a positional argument for the training config file path
    parser.add_argument('config', help='train config file path')
    # Add an optional argument for the work directory
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # Add a boolean flag to enable automatic-mixed-precision training
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training'
    )
    # Add a boolean flag to enable automatically scaling LR
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.'
    )
    # Add an optional argument to resume training from a checkpoint
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.'
    )
    # Add an optional argument to override settings in the config file
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.'
    )
    # Add an optional argument for the job launcher
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    # Add an optional argument for the local rank in distributed training
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # Parse the command-line arguments
    args = parser.parse_args()
    # Set the LOCAL_RANK environment variable if not already set
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    """
    Main function to run the detector training process.

    This function parses command-line arguments, loads the configuration,
    sets up the work directory, enables automatic mixed precision and LR scaling if specified,
    configures the resume options, builds the runner, and starts training.
    """
    # Parse the command-line arguments
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    # Load the configuration from the specified file
    cfg = Config.fromfile(args.config)
    # Set the launcher in the configuration
    cfg.launcher = args.launcher
    # Merge additional configuration options if provided
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    # Update the work directory based on the command-line argument or the configuration
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    # Enable automatic-mixed-precision training if specified
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    # Enable automatically scaling LR if specified
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    # Configure the resume options based on the command-line argument
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    # Build the runner based on the configuration
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    # Start the training process
    runner.train()


if __name__ == '__main__':
    # Call the main function if the script is run directly
    main()
