_base_ = 'mmdet::glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
# 30 is an empirical value, just set it to the maximum value
# without affecting the evaluation result
data_root = 'data/d3/'

model = dict(bbox_head=dict(type='GLIPHead'),
             test_cfg=dict(max_per_img=30))

val_dataloader = dict(batch_size=1)

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
                   'scale_factor', 'text', 'custom_entities', 'sent_ids'))
]

# -------------------------------------------------#
val_dataset_full = dict(
    type='DODDataset',
    data_root=data_root,
    ann_file='d3_json/d3_full_annotations.json',
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    pipeline=test_pipeline,
    test_mode=True,
    backend_args=None,
    return_classes=True)

val_evaluator_full = dict(
    type='RDODMetric',
    vis=True,
    save_root="visualization/dod_vis",
    ann_file=data_root + 'd3_json/d3_full_annotations.json')


# -------------------------------------------------#
datasets = [val_dataset_full]
dataset_prefixes = ['FULL']
metrics = [val_evaluator_full]

val_dataloader = dict(batch_size=1,
                      dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
