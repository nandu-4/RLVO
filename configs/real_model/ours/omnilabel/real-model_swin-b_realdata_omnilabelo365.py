# Base configuration file for the model
_base_ = '../real-model_swin-b_realdata.py'

# Set the chunked size to 1 and the maximum number of detections per image to 300
model = dict(test_cfg=dict(chunked_size=1, max_per_img=300))

# Number of worker processes for validation data loading
val_num_workers = 10
# Batch size for validation
val_batch_size = 1

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
        with_bbox=True, 
        with_label=False
    ),
    # Pack the detection inputs and include necessary metadata
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

# Specify the type of dataset to be used
dataset_type = 'OmniLabelDataset'
# Root directory for the dataset
data_root = 'data/'

# Backend arguments for data loading
backend_args = None

# Mapping for dataset paths
path_mapping = {'object365/': 'object365v2/object365/images/val/'}

# ---------------------OmniLabelO365---------------------------#
# Configure the validation data loader
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=val_num_workers,
    dataset=dict(
        _delete_=True, 
        # Specify the dataset type
        type='OmniLabelDataset',
        # Apply the path mapping
        path_mapping=path_mapping,
        # Annotation file for the validation dataset
        ann_file='data/omnilabel_val_v0.1.3/dataset_all_val_v0.1.3_object365.json',
        # Prefix for the image data
        data_prefix=dict(img='data/'),
        # Set the dataset to test mode
        test_mode=True,
        # Skip category processing
        skip_catg=True,
        # Apply the test pipeline to the dataset
        pipeline=test_pipeline,
        # Backend arguments for the dataset
        backend_args=None
    )
)
# Use the same configuration for the test data loader
test_dataloader = val_dataloader

# Configure the validation evaluator
val_evaluator = dict(
    _delete_=True,
    # Specify the evaluation metric type
    type='OmniLabelMetric',
    # Annotation file for the evaluation
    ann_file='data/omnilabel_val_v0.1.3/dataset_all_val_v0.1.3_object365.json',
    # Whether to only format the results without actual evaluation
    format_only=False
)
# Use the same configuration for the test evaluator
test_evaluator = val_evaluator

