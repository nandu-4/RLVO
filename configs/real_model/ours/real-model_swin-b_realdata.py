# Base configuration file for the model
_base_ = 'mmdet::mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

# ------------------------ Modify settings --------------------------

# URL to load the pre-trained model weights
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'

# Name of the language model to use
lang_model_name = 'bert-base-uncased'
# Number of positive expressions per instance
num_expressions_per_instance = 1
# Number of negative expressions
num_negative_expressions = 50
# Maximum number of tokens in the input
max_tokens = 512
# Learning rate for the optimizer
lr = 2 * 1e-6 
# Batch size for training
batch_size = 4
# Number of worker processes for data loading
num_workers = 8
# Number of GPUs to use for training
num_gpu = 8 * 2
# Maximum number of training epochs
max_epochs = 10
# Number of warm-up steps
warmup_step = 1000
# Sampling probabilities for different query types
sample_p = {
    "category": 0.5,
    "vlm_short": 0.2,
    "llm": 0.2,
    "vlm_long": 0.1
}

# Root directory for the data
data_root = "data"

# Directory for Object365 training images
o365_train_img_dir = 'object365/images/train/'
# Directory for OpenImage training images
oi_train_img_dir = 'openimage/train/'
# Directory for LVIS training images
lvis_train_img_dir = 'coco/train2017/'

# Path to the Object365 training annotation file
o365_train_ann_path = 'real-data/real-data-o365-base.jsonl'
# Path to the OpenImage training annotation file
oi_train_ann_path = 'real-data/real-data-openimage-base.jsonl'
# Path to the LVIS training annotation file
lvis_train_ann_path = 'real-data/real-data-lvis.jsonl'

# Number of worker processes for validation data loading
val_num_workers = 1
# Batch size for validation
val_batch_size = 1

# Path mapping for validation data
path_mapping = {'coco/': 'coco/val2017/'}

# --------------------------- model---------------------------
# Model configuration
model = dict(
    type="RealModel",
    # Backbone network configuration
    backbone=dict(
        # Delete the default backbone configuration
        _delete_=True,
        type='SwinTransformer',
        # Pretrained image size
        pretrain_img_size=384,
        # Embedding dimension
        embed_dims=128,
        # Number of layers in each stage
        depths=[2, 2, 18, 2],
        # Number of attention heads in each stage
        num_heads=[4, 8, 16, 32],
        # Window size for self-attention
        window_size=12,
        # MLP ratio
        mlp_ratio=4,
        # Bias for query, key, value projections
        qkv_bias=True,
        # Scale factor for query, key projections
        qk_scale=None,
        # Dropout rate
        drop_rate=0.,
        # Attention dropout rate
        attn_drop_rate=0.,
        # Drop path rate
        drop_path_rate=0.3,
        # Whether to use patch normalization
        patch_norm=True,
        # Output indices of the backbone
        out_indices=(1, 2, 3),
        # Whether to use checkpointing
        with_cp=True,
        # Convert weights from pre-trained models
        convert_weights=True,
        # Frozen stages of the backbone
        frozen_stages=-1,
        # Initialization configuration
        init_cfg=None
    ),
    # Language model configuration
    language_model=dict(max_tokens=max_tokens),
    # Neck network configuration
    neck=dict(in_channels=[256, 512, 1024]),
    # Bounding box head configuration
    bbox_head=dict(
        type='RealModelHead',
        # Configuration for contrastive learning
        contrastive_cfg=dict(max_text_len=max_tokens)
    ),
    # Test configuration
    test_cfg=dict(
        max_per_img=300,
        chunked_size=1,
    )
)

# --------------------------- train data---------------------------
# Training data pipeline configuration
train_pipeline = [
    # Randomly sample positive expressions
    dict(
        type='RandomSamplePosExpr',
        num_expressions_per_instance=num_expressions_per_instance,
        is_single=True,
        sample_p=sample_p
    ),
    # Load image from file
    dict(type='LoadImageFromFile', backend_args=None),
    # Load annotations
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    # Randomly flip the image
    dict(type='RandomFlip', prob=0.5),
    # Randomly choose a transformation
    dict(
        type='RandomChoice',
        transforms=[
            [
                # Randomly choose a scale for resizing
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)
                    ],
                    keep_ratio=True
                )
            ],
            [
                # Randomly choose a scale for resizing
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True
                ),
                # Randomly crop the image
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True
                ),
                # Randomly choose a scale for resizing
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)
                    ],
                    keep_ratio=True
                )
            ]
        ]
    ),
    # Filter annotations based on minimum bounding box size
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # Randomly sample negative expressions
    dict(
        type='RandomSampleNegExpr',
        tokenizer_name=lang_model_name,
        num_negative_expressions=num_negative_expressions,
        max_tokens=max_tokens
    ),
    # Pack the data for detection
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'tokens_positive'
        )
    )
]

# Object365 object detection dataset configuration
o365_od_dataset = dict(
    type='LODDatasetMultiSource',
    data_root=data_root,
    ann_file=o365_train_ann_path,
    data_prefix=dict(img=o365_train_img_dir),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None
)

# OpenImage object detection dataset configuration
oi_od_dataset = dict(
    type='LODDatasetMultiSource',
    data_root=data_root,
    ann_file=oi_train_ann_path,
    data_prefix=dict(img=oi_train_img_dir),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None
)

# LVIS object detection dataset configuration
lvis_od_dataset = dict(
    type='LODDatasetMultiSource',
    data_root=data_root,
    ann_file=lvis_train_ann_path,
    data_prefix=dict(img=lvis_train_img_dir),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None
)

# Training data loader configuration
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        # Delete the default sampler configuration
        _delete_=True,
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=[1, 1, 1]
    ),
    dataset=dict(
        datasets=[
            o365_od_dataset,
            oi_od_dataset,
            lvis_od_dataset
        ]
    )
)

# --------------------------- optimizer---------------------------
# Optimizer wrapper configuration
optim_wrapper = dict(
    # Delete the default optimizer wrapper configuration
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.0001
    ),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1)
        }
    )
)

# Learning policy configuration
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1
)

# Parameter scheduler configuration
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=warmup_step
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6, 8],
        gamma=0.1
    )
]

# Automatic learning rate scaling configuration
auto_scale_lr = dict(
    base_batch_size=32,
    enable=True
)
# Default hooks configuration
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=100
    )
)
# Log processor configuration
log_processor = dict(by_epoch=True)


# --------------------------- test data---------------------------
# Validation data pipeline configuration
omnilabel_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'
    ),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'
    ),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_label=False
    ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'description_ids'
        )
    )
]

# Define the validation dataset configuration for OmniLabel in COCO format
# The _delete_=True flag indicates that any existing configuration with the same name should be deleted
val_dataset_omnilabelcoco = dict(
    _delete_=True,
    # Specify the dataset type as OmniLabelDataset
    type='OmniLabelDataset',
    # Apply the path mapping defined earlier
    path_mapping=path_mapping,
    # Path to the annotation file for the validation dataset
    ann_file='data/omnilabel_val_v0.1.3/dataset_all_val_v0.1.3_coco.json',
    # Prefix for the image data directory
    data_prefix=dict(img='data/'),
    # Set the dataset to test mode
    test_mode=True,
    # Skip category-related processing
    skip_catg=True,
    # Use the predefined test pipeline for data processing
    pipeline=omnilabel_test_pipeline,
    # No backend-specific arguments
    backend_args=None
)

# Define the validation evaluator configuration for OmniLabel in COCO format
# The _delete_=True flag indicates that any existing configuration with the same name should be deleted
val_evaluator_omnilabelcoco = dict(
    _delete_=True,
    # Specify the evaluator type as OmniLabelMetric
    type='OmniLabelMetric',
    # Path to the annotation file for the validation dataset
    ann_file='data/omnilabel_val_v0.1.3/dataset_all_val_v0.1.3_coco.json',
    # Do not only format the results, perform actual evaluation
    format_only=False
)

# Define the validation data loader configuration
val_dataloader = dict(
    # Batch size for validation
    batch_size=val_batch_size,
    # Number of worker processes for data loading
    num_workers=val_num_workers,
    # Use the defined validation dataset
    dataset=val_dataset_omnilabelcoco
)
# The test data loader uses the same configuration as the validation data loader
test_dataloader = val_dataloader

# Use the OmniLabel evaluator for validation
val_evaluator = val_evaluator_omnilabelcoco
# Use the same evaluator for testing
test_evaluator = val_evaluator

# Configure the default hooks for the training process
default_hooks = dict(
    checkpoint=dict(
        # Maximum number of checkpoints to keep
        max_keep_ckpts=30,
        # Save the best model based on the specified metric
        save_best=['OmniLabel_COCO/omnilabel/bbox_AP_descr'],
        # The rule for determining the best model is greater value
        rule=['greater']
    )
)