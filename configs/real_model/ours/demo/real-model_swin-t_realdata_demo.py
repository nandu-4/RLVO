_base_ = '../real-model_swin-t_realdata.py'

model = dict(
    test_cfg=dict(
        max_per_img=300,
        chunked_size=1
    )
)

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    dataset=dict(_delete_=True,
                 dataset_type = 'CocoDataset',
                 data_root = 'data/coco/',
                 ann_file='annotations/instances_train2017.json',
                 data_prefix=dict(img='train2017/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=test_pipeline,
                 return_classes=True))
test_dataloader = val_dataloader