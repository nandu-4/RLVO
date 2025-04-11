# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import os.path as osp
from typing import List
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.transforms.text_transformers import clean_name
from omnilabeltools import OmniLabel


@DATASETS.register_module()
class OmniLabelDataset(BaseDetDataset):
    """
    A custom dataset class for working with OmniLabel data.

    This class inherits from BaseDetDataset and provides methods for loading and processing OmniLabel data.

    Args:
        path_mapping (dict): A dictionary for mapping paths in the data.
        skip_catg (bool): Whether to skip category labels.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.
    """
    OmniLabelAPI = OmniLabel

    def __init__(
        self,
        *args,
        path_mapping={},
        skip_catg=False,
        **kwargs
    ):
        self.path_mapping = path_mapping
        self.skip_catg = skip_catg
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """
        Load the data list from the annotation file.

        This method reads the annotation file, processes the data, and returns a list of dictionaries
        containing information about each image and its associated annotations.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary represents an image and its annotations.
        """
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.omnilabel = self.OmniLabelAPI(local_path)

        img_ids = self.omnilabel.image_ids
        data_list = []

        for img_id in img_ids:
            img_info = self.omnilabel.get_image_sample(img_id)
            ann_info = img_info.get('instances', [])

            file_name = img_info['file_name']
            for key, value in self.path_mapping.items():
                file_name = file_name.replace(key, value, 1)

            img_path = osp.join(self.data_prefix.get('img', ""), file_name)
            data_info = {}
            instances = []

            for i, ann in enumerate(ann_info):
                instance = {}

                if ann.get('ignore', False):
                    continue

                x1, y1, w, h = ann['bbox']

                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue

                bbox = [x1, y1, x1 + w, y1 + h]

                if ann.get('iscrowd', False):
                    instance['ignore_flag'] = 1
                else:
                    instance['ignore_flag'] = 0

                instance['bbox'] = bbox
                instances.append(instance)

            data_info['img_path'] = img_path
            data_info['img_id'] = img_info['id']
            data_info['custom_entities'] = False
            data_info['tokens_positive'] = -1

            text = []
            description_ids = []

            for labelspace in img_info["labelspace"]:
                if not self.skip_catg:
                    text.append(clean_name(labelspace['text']))
                    description_ids.append(labelspace['id'])
                else:
                    if labelspace["type"] != "C":
                        text.append(clean_name(labelspace['text']))
                        description_ids.append(labelspace['id'])

            data_info['text'] = text
            data_info['description_ids'] = description_ids
            data_info['instances'] = instances
            data_list.append(data_info)

        del self.omnilabel
        return data_list