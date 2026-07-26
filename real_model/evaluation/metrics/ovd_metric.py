# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

import tempfile
import numpy as np
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load

from mmdet.registry import METRICS
from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP

import datetime

def calculate_iou(gt_bbox, pred_bboxes):
    """
    Calculate the intersection ratio (IoU)
    """
    x11, y11, w1, h1  = gt_bbox
    x12 = x11 + w1
    y12 = y11 + h1
    
    x21 = pred_bboxes[:, 0]
    y21 = pred_bboxes[:, 1]
    x22 = pred_bboxes[:, 2]
    y22 = pred_bboxes[:, 3]
    
    w2 = x22 - x21
    h2 = y22 - y21

    xmin = np.maximum(x11, x21)
    ymin = np.maximum(y11, y21)
    xmax = np.minimum(x12, x22)
    ymax = np.minimum(y12, y22)

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    union = w1 * h1 + w2 * h2 - intersection

    # Handle cases where the denominator is zero
    iou = np.where(union == 0, 0, intersection / union)

    return iou

def nms_threaded(gt, prediction, iou_threshold, num_threads):
    gt_boxes = np.array([box['bbox'] for box in gt['anns']])


    keep_list = []
    remove_list = []
    for idx, i in enumerate(gt['anns']):
        gt_box = gt_boxes[idx]
        iou = calculate_iou(gt_box, prediction['bboxes'])
        indices =  np.where(iou > iou_threshold)[0].tolist()
        if indices:
            match_scores = prediction['scores'][indices]
            sorted_indices = np.argsort(match_scores)[::-1]

            final_indices = []
            for sort_idx in sorted_indices.tolist():
                final_indices.append(indices[sort_idx])

            if final_indices:
                keep_list.append(final_indices[0])
                remove_list += final_indices[1:]

    final_remove_list = []
    for i in remove_list:
        if i not in keep_list:
            final_remove_list.append(i)

    selected_boxes = []
    selected_scores = []
    selected_labels = []
    for idx, (bbox, score, label) in enumerate(zip(prediction["bboxes"],
                                                   prediction["scores"],
                                                   prediction["labels"])):
        if idx not in final_remove_list:
            selected_boxes.append(bbox)
            selected_scores.append(score)
            selected_labels.append(label)

    prediction["bboxes"] = selected_boxes
    prediction["scores"] = selected_scores
    prediction["labels"] = selected_labels
    return prediction

def cal_map(gt_data, pred_data):
    
    anno = COCO(gt_data)
    pred = anno.loadRes(pred_data)
    eval = COCOeval(anno, pred, 'bbox')

    eval.evaluate()
    eval.accumulate()
    eval.summarize()


@METRICS.register_module()
class OVDEvalMetric(BaseMetric):
    default_prefix: Optional[str] = 'OVDEval'
    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 format_only: bool = False,
                 iou_threshold = 0.5,
                 use_neg=False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 num_threads = 16,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")
                
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval
        
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        
        
        if ann_file is not None:
            with get_local_path(ann_file, 
                                backend_args=self.backend_args) as local_path:
                self._ovdeval_api = COCO(local_path)
        else:
            self._ovdeval_api = None

        self.iou_threshold = iou_threshold
        self.num_threads = num_threads
        self.use_neg = use_neg
        
        self._cat2id = {}
        for cat_id in self._ovdeval_api.getCatIds():
            cat_name = self._ovdeval_api.loadCats([cat_id])[0]['name'].replace('\"', '"')
            self._cat2id[cat_name] = cat_id
        
        self.img_ids = self._ovdeval_api.getImgIds()
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            if data_sample['img_id'] not in self.img_ids:
                continue
            
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            texts = data_sample["text"]

            result['labels'] = np.array([self._cat2id.get(texts[label], -1) for label in labels])
            
            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            # gt['text'] = data_sample['text']
            # gt['description_ids'] = data_sample['description_ids']
            if self._ovdeval_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            else:
                ann_ids = self._ovdeval_api.getAnnIds([data_sample['img_id']])
                gt['anns'] = self._ovdeval_api.loadAnns(ann_ids)
            
            if self.iou_threshold != 0:
                result = nms_threaded(gt, result, self.iou_threshold, self.num_threads)                
            self.results.append((gt, result))
    
    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
    
    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                x, y, xx, yy = bboxes[i]
                if int(x) - int(y) != 0 and int(xx) - int(yy) != 0:
                    data['image_id'] = image_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(scores[i])
                    data['category_id'] = label
                    if label != -1:
                        bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])
        return result_files
    
    
    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=[],
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        # split gt and prediction list
        
        if len(results) == 0:
            logger.error('The testing results of the whole dataset is empty.')
            return {}
        
        gts, preds = zip(*results)
        
        
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        
        if self._ovdeval_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._ovdeval_api = COCO(coco_json_path)
        
        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)
        
        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results
        
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            iou_type = 'bbox'
            
            # evaluate proposal, bbox and segm
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                ovdeval_dt = self._ovdeval_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break
            
            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._ovdeval_api, ovdeval_dt, iou_type)
            else:
                coco_eval = COCOeval(self._ovdeval_api, ovdeval_dt, iou_type)

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[coco_metric_names[metric_item]]
                eval_results[key] = float(f'{round(val, 3)}')

            ap = coco_eval.stats[:6]
            logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                        f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
        
        