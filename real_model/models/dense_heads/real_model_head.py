# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

@MODELS.register_module()
class RealModelHead(GroundingDINOHead):
    """
    RealModelHead class inherits from GroundingDINOHead.
    This class is used to transform the features extracted from the head into bounding box results for a single image.
    """
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """
        Transform a single image's features extracted from the head into bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer for each image. 
                                Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer for each image, 
                                with coordinate format (cx, cy, w, h) and shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image after the post process.
        """
        # Ensure the number of class scores and bounding box predictions are the same
        assert len(cls_score) == len(bbox_pred)  # num_queries
        
        # Get the image shape from the image meta information
        img_shape = img_meta['img_shape']
        # Determine the maximum number of detections per image
        max_per_img = int(min(self.test_cfg.get('max_per_img', len(cls_score)), len(cls_score)))
        # If token positive maps are provided
        if token_positive_maps is not None:
            # Convert grounding scores to classification scores
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            # Get the top-k scores and their corresponding indices
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            # Get the number of classes
            num_classes = cls_score.shape[-1]
            # Determine the class labels for the top-k detections
            det_labels = indexes % num_classes
            # Determine the indices of the bounding box predictions for the top-k detections
            bbox_index = indexes // num_classes
            # Select the corresponding bounding box predictions
            bbox_pred = bbox_pred[bbox_index]
        else:
            # Apply sigmoid function to the class scores
            cls_score = cls_score.sigmoid()
            # Get the maximum score for each query
            scores, _ = cls_score.max(-1)
            # Get the top-k scores and their corresponding indices
            scores, indexes = scores.topk(max_per_img)
            # Select the corresponding bounding box predictions
            bbox_pred = bbox_pred[indexes]
            # Initialize the class labels as zeros
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        # Convert bounding box coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        # Scale the x-coordinates to the image width
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        # Scale the y-coordinates to the image height
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        # Clamp the x-coordinates to the image width
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        # Clamp the y-coordinates to the image height
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        # If rescale is True, rescale the bounding boxes to the original image size
        if rescale:
            # Ensure the scale factor is provided in the image meta information
            assert img_meta.get('scale_factor') is not None
            # Rescale the bounding boxes
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        # Create an InstanceData object to store the detection results
        results = InstanceData()
        # Store the bounding boxes in the results
        results.bboxes = det_bboxes
        # Store the scores in the results
        results.scores = scores
        # Store the class labels in the results
        results.labels = det_labels
        # Return the detection results
        return results