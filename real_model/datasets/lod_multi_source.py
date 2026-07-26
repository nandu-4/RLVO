# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import json
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class LODDatasetMultiSource(BaseDetDataset):
    """object detection and visual grounding dataset."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the LODDatasetMultiSource class.

        Args:
            *args: Variable length argument list passed to the parent class.
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """
        Load data list from the annotation file and preprocess it.

        Returns:
            List[dict]: A list of dictionaries, each containing image and pair information.
        """
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in data_list:
            pairs = data.get('pairs')
            if pairs is not None:
                data_info = {
                    'img_path': osp.join(self.data_prefix['img'], data['filename']),
                    'height': data['height'],
                    'width': data['width'],
                    'pairs': pairs
                }
                pairs_num = sum(len(pairs[pair_name]) for pair_name in pairs)
                if pairs_num > 0:
                    out_data_list.append(data_info)

        del data_list
        return out_data_list

    def __getitem__(self, idx: int) -> dict:
        """
        Get the idx-th image and data information after applying the pipeline.
        If the dataset is not fully initialized, call `full_init`.
        During training, retry fetching data if the result is invalid.

        Args:
            idx (int): Index of the data in self.data_list.

        Returns:
            dict: Image and data information after the pipeline.
        """
        if not self._fully_initialized:
            # print_log(
            #     'Please call `full_init()` method manually to accelerate '
            #     'the speed.',
            #     logger='current',
            #     level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipeline should not get `None` data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None or data['data_samples'].metainfo['text'] == "":
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')