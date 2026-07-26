# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

# Import necessary modules
import os.path as osp
from typing import List
from pathlib import Path

from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.api_wrappers.coco_api import COCO
from mmengine.fileio import get_local_path


@DATASETS.register_module()
class OVDEvalDataset(BaseDetDataset):
    """
    A custom dataset class for OVD evaluation.

    This class inherits from BaseDetDataset and provides methods for loading and processing OVD evaluation data.

    Attributes:
        OVDEvalAPI (type): The API class used for working with the OVD evaluation data.
        use_neg (bool): Whether to use negative samples in the dataset.
    """
    # Set the OVDEvalAPI to the COCO class
    OVDEvalAPI = COCO

    def __init__(self, use_neg=True, *args, **kwargs) -> None:
        """
        Initialize the OVDEvalDataset class.

        Args:
            use_neg (bool, optional): Whether to use negative samples. Defaults to True.
            *args: Additional positional arguments passed to the base class.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        # Set the flag to use negative samples
        self.use_neg = use_neg
        # Call the constructor of the base class
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """
        Load the data list from the annotation file.

        This method reads the annotation file, processes the data, and returns a list of dictionaries
        containing information about each image and its associated annotations.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary represents an image and its annotations.
        """
        # Get the local path of the annotation file
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            # Initialize the OVDEval API with the local path of the annotation file
            self.ovdeval = self.OVDEvalAPI(local_path)

        # Get the dataset name by removing the .json extension from the annotation file name
        dataset_name = Path(self.ann_file).name.replace(".json", "")
        # Get the list of image IDs from the OVDEval API
        img_ids = self.ovdeval.getImgIds()
        # Initialize a dictionary to map category names to category IDs
        cat2id = {}

        # Populate the cat2id dictionary
        for cat_id in self.ovdeval.getCatIds():
            cat_name = self.ovdeval.loadCats([cat_id])[0]['name']
            cat2id[cat_name] = cat_id

        # Initialize an empty list to store data information for each image
        data_list = []

        # Iterate over each image ID
        for img_id in img_ids:
            # Get the image information for the current image ID
            img_info = self.ovdeval.loadImgs([img_id])[0]

            # Extract the file name of the current image
            file_name = img_info['file_name']
            # Construct the full path to the image file
            img_path = osp.join(self.data_prefix.get('img', ""), file_name)

            # Initialize a dictionary to store data information for the current image
            data_info = {
                'img_path': img_path,
                'img_id': img_id,
                'height': img_info['height'],
                'width': img_info['width'],
                'instances': []
            }

            # Determine the text information based on the dataset name and use_neg flag
            if dataset_name in ['coco', 'celebrity', 'logo', 'landmark']:
                data_info['text'] = [
                    self.ovdeval.loadCats([cat_id])[0]['name']
                    for cat_id in self.ovdeval.getCatIds()
                ]
            else:
                if self.use_neg:
                    data_info['text'] = [
                        t for t in img_info["text"] + img_info["neg_text"]
                    ]
                else:
                    data_info['text'] = [t for t in img_info["text"]]

            # Replace escaped double quotes with regular double quotes in the text information
            data_info['text'] = [
                text.replace('\"', '"') for text in data_info['text']
            ]

            # Set additional information in the data_info dictionary
            data_info['custom_entities'] = False
            data_info['tokens_positive'] = -1

            # Add the data information for the current image to the data list
            data_list.append(data_info)

        # Delete the OVDEval API object to free up memory
        del self.ovdeval

        # Return the list of data information for all images
        return data_list