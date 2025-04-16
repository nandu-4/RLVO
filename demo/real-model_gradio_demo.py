# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import torch
import gradio as gr
from mmengine.logging import MMLogger
from argparse import ArgumentParser
from mmdet.apis import DetInferencer
from real_model import *

# Initialize logger
logger = MMLogger('mmdetection', logger_name='mmdet')

# Check for available GPUs
if torch.cuda.is_available():
    gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    logger.info(f'Available GPUs: {len(gpus)}')
else:
    gpus = None
    logger.info('No available GPU.')

# Function to get a free device
def get_free_device():
    if gpus is None:
        return torch.device('cpu')
    if hasattr(torch.cuda, 'mem_get_info'):
        free = [torch.cuda.mem_get_info(gpu)[0] for gpu in gpus]
        select = max(zip(free, range(len(free))))[1]
    else:
        import random
        select = random.randint(0, len(gpus) - 1)
    return gpus[select]

# RealModelDemo class for Gradio-based interface
class RealModelDemo:
    """
    Interactive Gradio-based interface for Real-Model object detection inference.

    Args:
        config (str): Path to the configuration file.
        checkpoint (str): Path to the checkpoint file.
        score_thre (float, optional): Score threshold for inference, default is 0.3.
        device (str or torch.device, optional): Device used for inference, default is 'cuda:0'.
    """
    def __init__(self, config, checkpoint, score_thre=0.3, device='cuda:0') -> None:
        self.device = device or get_free_device()
        self.inferencer = DetInferencer(config, checkpoint, device=self.device)
        self.score_thre = score_thre
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label='Image', source='upload', elem_classes='input_image',
                    type='filepath', interactive=True, tool='editor'
                )
                text_input = gr.Textbox(
                    label='Text Prompt', elem_classes='input_text', interactive=True
                )
                score_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=self.score_thre, step=0.01,
                    label='Score Threshold', interactive=True
                )
                output = gr.Image(
                    label='Result', source='upload', interactive=False, elem_classes='result'
                )
                run_button = gr.Button('Run', elem_classes='run_button')
                run_button.click(
                    self.inference, inputs=[image_input, text_input, score_slider], outputs=output
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input, text_input],
                samples=[
                    ['demo/demo_images/demo.jpg', 'man riding a carriage'],
                    ['demo/demo_images/006_50930592.jpg', 'woman in wedding dress next to a man in suit'],
                    ['demo/demo_images/000000060823.jpg', 'cows that are laid down'],
                    ['demo/demo_images/000000081988.jpg', 'these two people each have a pink surfboard']
                ]
            )
            example_images.click(
                fn=self.update, inputs=example_images, outputs=[image_input, text_input]
            )

    def update(self, example):
        return gr.Image.update(value=example[0]), gr.Textbox.update(value=example[1])

    def inference(self, image, text, score_thre):
        results_dict = self.inferencer(
            image, texts=text, pred_score_thr=score_thre, custom_entities=False,
            tokens_positive=-1, return_vis=True, no_save_vis=True
        )
        return results_dict['visualization'][0]

# Function to parse command-line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default=None, help='Device used for inference')
    parser.add_argument('--server_name', default='0.0.0.0', help='Gradio server name (default: 0.0.0.0)')
    parser.add_argument('--server_port', type=int, default=7860, help='Gradio server port (default: 7860)')
    parser.add_argument('--score_thre', type=float, default=0.1, help='Score threshold for inference (default: 0.3)')
    parser.add_argument('--share', action='store_true', help='Enable sharing the Gradio app (default: False)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for Gradio (default: False)')
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    args = parse_args()
    title = 'Real-Model Inference Demo'
    DESCRIPTION = '''# <div align="center">Real-Model Inference Demo</div>
    
    #### This is an official demo for Real-Model.  
    Reference to MMDetection.  
    '''
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(DESCRIPTION)
        RealModelDemo(
            args.config, args.checkpoint, score_thre=args.score_thre, device=args.device
        )
    demo.launch(
        server_name=args.server_name, server_port=args.server_port,
        share=args.share, debug=args.debug
    )