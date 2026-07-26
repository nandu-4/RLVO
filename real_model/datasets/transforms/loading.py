# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadLODAnnotations(LoadAnnotations):
    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['postive_expression_id'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.