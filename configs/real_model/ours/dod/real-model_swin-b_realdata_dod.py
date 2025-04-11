# Base configuration file for the model
_base_ = '../real-model_swin-b_realdata.py'

# Root directory for the dataset
data_root = 'data/d3/'

# Set the chunked size to 1 and the maximum number of detections per image to 300
model = dict(test_cfg=dict(chunked_size=1))

# Define the test data processing pipeline
test_pipeline = [
    # Load an image from a file using the specified backend
    dict(
        type='LoadImageFromFile', 
        backend_args=None,
        imdecode_backend='pillow'
    ),
    # Resize the image to a fixed scale while maintaining the aspect ratio
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'
    ),
    # Load annotations for the image, including bounding boxes but not labels
    dict(
        type='LoadAnnotations', 
        with_bbox=True
    ),
    # Pack the detection inputs and include necessary metadata
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'text', 'custom_entities', 'sent_ids'
        )
    )
]

# ---------------------DOD Full Dataset---------------------------#
# Full validation dataset configuration
val_dataset_full = dict(
    type='DODDataset',
    data_root=data_root,
    # Path to the annotation file
    ann_file='d3_json/d3_full_annotations.json',
    # Data prefix, specifying the relative paths of images and annotation files
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    # Use the test data processing pipeline
    pipeline=test_pipeline,
    # Test mode
    test_mode=True,
    backend_args=None,
    # Return class information
    return_classes=True
)

# Full validation dataset evaluator configuration
val_evaluator_full = dict(
    type='DODCocoMetric',
    # Path to the annotation file
    ann_file=data_root + 'd3_json/d3_full_annotations.json'
)

# ---------------------DOD Pres Dataset---------------------------#
# Pres validation dataset configuration
val_dataset_pres = dict(
    type='DODDataset',
    data_root=data_root,
    # Path to the annotation file
    ann_file='d3_json/d3_pres_annotations.json',
    # Data prefix, specifying the relative paths of images and annotation files
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    # Use the test data processing pipeline
    pipeline=test_pipeline,
    # Test mode
    test_mode=True,
    backend_args=None,
    # Return class information
    return_classes=True
)

# Pres validation dataset evaluator configuration
val_evaluator_pres = dict(
    type='DODCocoMetric',
    # Path to the annotation file
    ann_file=data_root + 'd3_json/d3_pres_annotations.json'
)

# ---------------------DOD Abs Dataset---------------------------#
# Abs validation dataset configuration
val_dataset_abs = dict(
    type='DODDataset',
    data_root=data_root,
    # Path to the annotation file
    ann_file='d3_json/d3_abs_annotations.json',
    # Data prefix, specifying the relative paths of images and annotation files
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    # Use the test data processing pipeline
    pipeline=test_pipeline,
    # Test mode
    test_mode=True,
    backend_args=None,
    # Return class information
    return_classes=True
)

# Abs validation dataset evaluator configuration
val_evaluator_abs = dict(
    type='DODCocoMetric',
    # Path to the annotation file
    ann_file=data_root + 'd3_json/d3_abs_annotations.json'
)

# ---------------------Combined Validation Configuration---------------------------#
# Combine all validation datasets
datasets = [val_dataset_full, val_dataset_pres, val_dataset_abs]
# List of dataset prefixes
dataset_prefixes = ['FULL', 'PRES', 'ABS']
# List of evaluators
metrics = [val_evaluator_full, val_evaluator_pres, val_evaluator_abs]

# Validation data loader configuration, using the combined dataset
val_dataloader = dict(
    dataset=dict(
        _delete_=True, 
        type='ConcatDataset', 
        datasets=datasets
    )
)
# Use the same configuration for the test data loader
test_dataloader = val_dataloader

# Validation evaluator configuration, using a multi-dataset evaluator
val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes
)
# Test evaluator uses the same configuration as the validation evaluator
test_evaluator = val_evaluator
