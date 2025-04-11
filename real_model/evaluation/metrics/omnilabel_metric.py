# Refer to MMDetection
# Copyright (c) VCIP - NKU. All rights reserved.

import os
import time
import datetime
import tempfile
import numpy as np
import os.path as osp
from collections import OrderedDict
from omnilabeltools import OmniLabel, OmniLabelEval
from typing import Dict, List, Optional, Sequence, Union

from mmdet.registry import METRICS
from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load


@METRICS.register_module()
class OmniLabelMetric(BaseMetric):
    """
    OmniLabelMetric class is used to evaluate object detection results based on the OmniLabel dataset.
    It inherits from the BaseMetric class and overrides methods to calculate metrics such as AP, AR, etc.
    """
    # Set the default prefix for the metric
    default_prefix: Optional[str] = 'omnilabel'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 score_thre: float = 0,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 use_mp_eval: bool = False) -> None:
        """
        Initialize the OmniLabelMetric class.

        Args:
            ann_file (Optional[str]): Path to the annotation file. Defaults to None.
            metric (Union[str, List[str]]): Metrics to be evaluated. Defaults to 'bbox'.
            score_thre (float): Score threshold for filtering predictions. Defaults to 0.
            format_only (bool): Whether to only format the results without actual evaluation. Defaults to False.
            outfile_prefix (Optional[str]): Prefix for the output files. Defaults to None.
            backend_args (dict): Arguments for the data backend. Defaults to None.
            collect_device (str): Device for collecting results. Defaults to 'cpu'.
            prefix (Optional[str]): Prefix for the metric names. Defaults to None.
            use_mp_eval (bool): Whether to use multi - processing for evaluation. Defaults to False.
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = [metric] if isinstance(metric, str) else metric
        allowed_metrics = ['bbox']
        for m in self.metrics:
            if m not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {m}.")

        self.use_mp_eval = use_mp_eval
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, (
                'outfile_prefix must not be None when format_only is True, '
                'otherwise the result files will be saved to a temp directory '
                'which will be cleaned up at the end.'
            )
        self.outfile_prefix = outfile_prefix
        self.backend_args = backend_args
        if ann_file is not None:
            with get_local_path(ann_file, backend_args=self.backend_args) as local_path:
                self._omnilabel_api = OmniLabel(local_path)
        else:
            self._omnilabel_api = None
        self.score_thre = score_thre

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = {
                'img_id': data_sample['img_id'],
                'bboxes': data_sample['pred_instances']['bboxes'].cpu().numpy(),
                'scores': data_sample['pred_instances']['scores'].cpu().numpy(),
                'labels': data_sample['pred_instances']['labels'].cpu().numpy()
            }
            gt = {
                'width': data_sample['ori_shape'][1],
                'height': data_sample['ori_shape'][0],
                'img_id': data_sample['img_id'],
                'text': data_sample['text'],
                'description_ids': data_sample['description_ids']
            }
            self.results.append((gt, result))

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """
        Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """
        _bbox = bbox.tolist()
        return [_bbox[0], _bbox[1], _bbox[2] - _bbox[0], _bbox[3] - _bbox[1]]

    def results2json(self, gts, results: Sequence[dict], outfile_prefix: str) -> dict:
        """
        Dump the detection results to a COCO style json file.

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
        for idx, (gt, result) in enumerate(zip(gts, results)):
            image_id = result.get('img_id', idx)
            labels = result['labels'].reshape(-1, 1)
            bboxes = result['bboxes']
            scores = result['scores'].reshape(-1, 1)
            for i, bbox in enumerate(bboxes):
                data = {
                    'image_id': image_id,
                    'bbox': self.xyxy2xywh(bbox),
                    'scores': [],
                    'description_ids': [],
                    'text': gt['text']
                }
                for label, score in zip(labels[i], scores[i]):
                    if score >= self.score_thre:
                        data['description_ids'].append(gt['description_ids'][label])
                        data['scores'].append(score)
                if data['description_ids']:
                    bbox_json_results.append(data)

        result_files = {}
        result_files['bbox'] = os.path.join(outfile_prefix, f'{datetime.datetime.now().timestamp()}.bbox.json')
        dump(bbox_json_results, result_files['bbox'])
        return result_files

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        result_files = self.results2json(gts, preds, outfile_prefix)
        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        start_time = time.time()
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                coco_dt = self._omnilabel_api.load_res(predictions)
            except IndexError:
                logger.error('The testing results of the whole dataset is empty.')
                break

            omnilabel_eval = OmniLabelEval(self._omnilabel_api, coco_dt)
            omnilabel_eval.evaluate()
            omnilabel_eval.accumulate()
            omnilabel_eval.summarize()
            print(f"Time: {(time.time() - start_time) / 60} min")

            metric_items = [
                'AP', 'AP_categ', 'AP_descr', 'AP_descr_p', 'AP_descr_s',
                'AP_descr_m', 'AP_descr_l', 'AP_descr_0.5', 'AP_descr_0.75',
                'AP_categ_0.5', 'AP_categ_0.75', 'AR_descr', 'AR_categ'
            ]
            for i, metric_item in enumerate(metric_items):
                key = f'{metric}_{metric_item}'
                val = omnilabel_eval.stats[i]
                eval_results[key] = float(f'{round(val, 3)}')

            ap = omnilabel_eval.stats
            logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f} {ap[7]:.3f} {ap[8]:.3f} {ap[9]:.3f} '
                        f'{ap[10]:.3f} {ap[11]:.3f} {ap[12]:.3f} ')

        if tmp_dir is not None and not self.format_only:
            tmp_dir.cleanup()
        return eval_results