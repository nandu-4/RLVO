# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import re
import json
import torch
import bisect
import random
import numpy as np
from collections import OrderedDict

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes
from mmcv.transforms import BaseTransform
from mmengine.fileio import get_local_path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

BACKGROUND_STR = ""
BACKGROUND_LABEL = -1


def clean_name(name):
    """
    Clean the input name by removing parentheses, underscores, extra spaces, and dots, and convert to lowercase.

    Args:
        name (str): The input name to be cleaned.

    Returns:
        str: The cleaned name.
    """
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = re.sub(r'\.', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(
    gt_bboxes,
    gt_labels,
    gt_ignore_flags,
    text,
    tokenizer,
    max_tokens
):
    """
    Check if the positive labels exceed the maximum token limit.

    Args:
        gt_bboxes (np.ndarray): Ground truth bounding boxes.
        gt_labels (np.ndarray): Ground truth labels.
        gt_ignore_flags (np.ndarray): Ground truth ignore flags.
        text (list): List of text corresponding to the labels.
        tokenizer: Tokenizer for text processing.
        max_tokens (int): Maximum number of tokens allowed.

    Returns:
        tuple: A tuple containing the kept bounding boxes, labels, ignore flags, and the length of the positive caption.
    """
    num_expressions_per_instance = gt_labels.shape[1]
    positive_label_list = np.unique(gt_labels).tolist()
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):
        if label == BACKGROUND_LABEL:
            continue
        label_text = clean_name(text[label]) + '. '
        tokenized = tokenizer.tokenize(label_text)
        length += len(tokenized)
        if length >= max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i, gt_label in enumerate(gt_labels):
        keep_gt_label = []
        for label in gt_label:
            if label in kept_lables:
                keep_gt_label.append(label)
        if len(keep_gt_label) > 0:
            keep_gt_label += [BACKGROUND_LABEL] * max(
                num_expressions_per_instance - len(keep_gt_label), 0
            )
            keep_gt_label = np.array(keep_gt_label, dtype=np.long)
            keep_gt_labels.append(keep_gt_label[None, :])
            keep_box_index.append(i)

    keep_gt_bboxes = gt_bboxes[keep_box_index]
    keep_gt_ignore_flags = gt_ignore_flags[keep_box_index]
    if len(keep_gt_labels) > 0:
        keep_gt_labels = np.concatenate(keep_gt_labels, axis=0)
    else:
        keep_gt_labels = np.empty((0, 1), dtype=np.long)

    return keep_gt_bboxes, keep_gt_labels, keep_gt_ignore_flags, length


def generate_senetence_given_labels(
    positive_label_list,
    negative_label_list,
    total_expressions
):
    """
    Generate a sentence given the positive and negative labels.

    Args:
        positive_label_list (list): List of positive labels.
        negative_label_list (list): List of negative labels.
        total_expressions (dict): Dictionary of all expressions.

    Returns:
        tuple: A tuple containing the label to positions dictionary, the generated sentence, and the label remap dictionary.
    """
    label_to_positions = {}
    label_list = positive_label_list + negative_label_list
    random.shuffle(label_list)

    pheso_caption = ''
    label_remap_dict = {}

    for index, label in enumerate(label_list):
        start_index = len(pheso_caption)
        pheso_caption += clean_name(total_expressions[label])
        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[label] = index
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplePosExpr(BaseTransform):
    """
    A custom transform class for randomly sampling positive expressions.

    Args:
        seed (int): Random seed for reproducibility.
        num_expressions_per_instance (int): Number of expressions per instance.
        use_catg (bool): Whether to use category information.
        use_origin (bool): Whether to use original positive expressions.
        sample_p (dict): Sampling probabilities for different query sources.
        **kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        seed=0,
        num_expressions_per_instance=1,
        use_catg=False,
        use_origin=False,
        sample_p={"category": 0.5, "shikra": 0.3, "llm": 0.1, "llava": 0.1},
        **kwargs
    ):
        self.seed = seed
        self.num_expressions_per_instance = num_expressions_per_instance
        self.use_catg = use_catg
        self.use_origin = use_origin
        self.sample_p = sample_p
        self.expr_names = list(sample_p.keys())
        expr_ps = [sample_p[key] for key in self.expr_names]
        self.expr_culp = [sum(expr_ps[:i]) for i in range(1, len(expr_ps))]

    def transform(self, results: dict) -> dict:
        """
        Apply the transformation to the input results.

        Args:
            results (dict): Input results dictionary.

        Returns:
            dict: Transformed results dictionary.
        """
        np.random.seed(self.seed)

        pair_infos = results["pairs"]
        is_multi = False
        expr_source = "other"

        if isinstance(pair_infos, dict):
            p = random.random()
            expr_index = bisect.bisect(self.expr_culp, p)
            expr_source = self.expr_names[expr_index]
            if "_" in expr_source:
                expr_source, is_multi = expr_source.split("_")
                is_multi = is_multi == "multi"
            pair_infos = pair_infos.get(expr_source, [])

        if self.use_catg:
            expr_source = "category"

        total_postive_expressions = OrderedDict()
        total_negative_expressions = []
        instance_list = []

        for pair_info in pair_infos:
            bboxes = pair_info['bboxes']
            category = pair_info['category']
            relation = pair_info.get("relation", "single")

            if is_multi:
                if relation != "multiple":
                    continue
                origin_postive_expressions = [
                    p for p in pair_info['postive_expressions'] if p != ""
                ]
                if len(origin_postive_expressions) == 0:
                    continue
                postive_expressions = np.random.choice(
                    origin_postive_expressions,
                    size=self.num_expressions_per_instance
                ).tolist()
            else:
                if relation != "single":
                    continue
                if expr_source == "category":
                    postive_expressions = [category]
                else:
                    if self.use_origin:
                        origin_postive_expressions = pair_info['origin_postive_expressions']
                    else:
                        origin_postive_expressions = pair_info['postive_expressions']
                    origin_postive_expressions = [
                        p for p in origin_postive_expressions if p != ""
                    ]
                    if len(origin_postive_expressions) == 0:
                        continue
                    postive_expressions = np.random.choice(
                        origin_postive_expressions,
                        size=self.num_expressions_per_instance
                    ).tolist()

            for p in postive_expressions:
                if p not in total_postive_expressions:
                    total_postive_expressions[p] = len(total_postive_expressions)

            negative_expressions = pair_info['negative_expressions']
            for negative_query in negative_expressions:
                if negative_query not in total_negative_expressions:
                    total_negative_expressions.append(negative_query)

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                inter_w = max(0, min(x2, results['width']) - max(x1, 0))
                inter_h = max(0, min(y2, results['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                instance = {
                    'ignore_flag': 0,
                    'bbox': bbox,
                    'bbox_label': [total_postive_expressions[p] for p in postive_expressions]
                }
                instance['bbox_label'] += [-1] * max(
                    self.num_expressions_per_instance - len(instance['bbox_label']), 0
                )
                instance_list.append(instance)

        results["total_postive_expressions"] = list(total_postive_expressions.keys())
        results["total_negative_expressions"] = total_negative_expressions
        results["instances"] = instance_list
        results['expr_source'] = expr_source
        return results


@TRANSFORMS.register_module()
class RandomSampleNegExpr(BaseTransform):
    """
    A custom transform class for randomly sampling negative expressions.

    Args:
        tokenizer_name (str): Name of the tokenizer to use.
        num_negative_expressions (int): Number of negative expressions to sample.
        negative_paths (list or str): Paths to the negative query files.
        max_tokens (int): Maximum number of tokens allowed.
        backend_args (dict): Backend arguments for file handling.
    """
    def __init__(
        self,
        tokenizer_name,
        num_negative_expressions=85,
        negative_paths=None,
        max_tokens=256,
        backend_args=None
    ):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.'
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=False
        )
        self.num_negative_expressions = num_negative_expressions
        self.max_tokens = max_tokens
        self.negative_expressions = []

        if negative_paths is not None:
            if isinstance(negative_paths, str):
                negative_paths = [negative_paths]
            for negative_path in negative_paths:
                with get_local_path(
                    negative_path, backend_args=backend_args
                ) as local_path:
                    with open(local_path, 'r') as f:
                        self.negative_expressions += json.load(f)

    def transform(self, results: dict) -> dict:
        """
        Apply the transformation to the input results.

        Args:
            results (dict): Input results dictionary.

        Returns:
            dict: Transformed results dictionary.
        """
        return self.lod_aug(results)

    def lod_aug(self, results):
        """
        Perform the LOD augmentation on the input results.

        Args:
            results (dict): Input results dictionary.

        Returns:
            dict: Augmented results dictionary.
        """
        gt_bboxes = results['gt_bboxes']
        if len(gt_bboxes) == 0:
            results['gt_bboxes_labels'] = gt_bboxes.new_zeros((0), dtype=torch.long)
            results['text'] = ""
            results['tokens_positive'] = {}
            return results

        gt_ignore_flags = results['gt_ignore_flags']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        total_postive_expressions = results["total_postive_expressions"]
        total_negative_expressions = results["total_negative_expressions"] + self.negative_expressions
        gt_labels = results['gt_bboxes_labels']

        original_box_num = len(gt_labels)
        gt_bboxes, gt_labels, gt_ignore_flags, positive_caption_length = check_for_positive_overflow(
            gt_bboxes,
            gt_labels,
            gt_ignore_flags,
            total_postive_expressions,
            self.tokenizer,
            self.max_tokens
        )

        if len(gt_bboxes) < original_box_num:
            print(
                'WARNING: removed {} boxes due to positive caption overflow'.format(
                    original_box_num - len(gt_bboxes)
                )
            )

        to_num_negative_expressions = self.num_negative_expressions - len(total_negative_expressions)

        if to_num_negative_expressions < 0:
            random.shuffle(total_negative_expressions)
            total_negative_expressions = total_negative_expressions[:self.num_negative_expressions]

        negative_max_length = self.max_tokens - positive_caption_length
        total_expressions = {}

        for i, negative_query in enumerate(total_negative_expressions):
            query = clean_name(negative_query) + '. '
            tokenized = self.tokenizer.tokenize(query)
            negative_max_length -= len(tokenized)
            if negative_max_length > 1:
                total_expressions[f"n_{i}"] = negative_query
            else:
                break

        positive_label_list = np.unique(gt_labels)
        positive_label_list = positive_label_list[
            positive_label_list != BACKGROUND_LABEL
        ].tolist()
        negative_label_list = list(total_expressions.keys())

        for positive_label in positive_label_list:
            total_expressions[positive_label] = total_postive_expressions[positive_label]

        label_to_positions, pheso_caption, label_remap_dict = generate_senetence_given_labels(
            positive_label_list,
            negative_label_list,
            total_expressions
        )
        label_remap_dict[BACKGROUND_LABEL] = BACKGROUND_LABEL

        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_ignore_flags'] = gt_ignore_flags
        results['gt_bboxes_labels'] = gt_labels[:, 0]
        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions
        return results

