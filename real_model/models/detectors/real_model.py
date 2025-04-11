# Refer to MMDetection
# Copyright (c) VCIP-NKU. All rights reserved.

# Import the copy module for deep copying objects
import copy
# Import the warnings module to handle warnings
import warnings

# Import the MODELS registry from mmdet
from mmdet.registry import MODELS
# Import the GroundingDINO detector class from mmdet
from mmdet.models.detectors import GroundingDINO

# Register the RealModel class in the MODELS registry
@MODELS.register_module()
class RealModel(GroundingDINO):
    """
    The RealModel class inherits from GroundingDINO and is used for object detection and recognition tasks.

    This class overrides the predict method to support custom text prompts and entity recognition.
    """
    def predict(self, 
                batch_inputs, 
                batch_data_samples, 
                rescale: bool = True):
        """
        Predict object detection results for a batch of inputs.

        Args:
            batch_inputs (list): Batch of input image data.
            batch_data_samples (list): Batch of data samples, including text prompts and entity information.
            rescale (bool, optional): Whether to rescale the prediction results. Defaults to True.

        Returns:
            list: Batch of data samples, including predicted instance information.
        """
        # Initialize lists to store text prompts, enhanced text prompts, and positive tokens
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        # Iterate through the batch of data samples to extract text prompts, enhanced text prompts, and positive tokens
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            # Check if the data sample contains a caption prompt
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            # Get the positive tokens from the data sample, default to None if not present
            tokens_positives.append(data_samples.get('tokens_positive', None))

        # Check if there are custom entities
        if 'custom_entities' in batch_data_samples[0]:
            # Assume that the `custom_entities` flag is always the same within a batch. Used for single-image inference.
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        # If all text prompts are the same, avoid redundant calculations
        if len(text_prompts) == 1:
            # All text prompts are the same, so there's no need to compute them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], 
                    custom_entities, 
                    enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            # Compute positive token maps and prompts for each text prompt
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        # Unpack positive token maps, text prompts, and entities
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # Extract visual features from the images
        visual_feats = self.extract_feat(batch_inputs)

        # Check if the first text prompt is a list, indicating chunked text prompts
        if isinstance(text_prompts[0], list):
            # Chunked text prompts, only support batch size of 1
            assert len(batch_inputs) == 1
            # Initialize a counter for label indexing
            count = 0
            # Initialize a list to store the original prediction results
            origin_results_list = []

            # Flatten the entities list
            entities = [[item for lst in entities[0] for item in lst]]

            # Process chunked text prompts
            for b in range(len(text_prompts[0])):
                # Select the current text prompt
                text_prompts_once = [text_prompts[0][b]]
                # Select the current token positive map
                token_positive_maps_once = token_positive_maps[0][b]
                # Extract text features
                text_dict = self.language_model(text_prompts_once)
                # Text feature mapping layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])
                # Assign the current token positive map to the data sample
                batch_data_samples[0].token_positive_map = token_positive_maps_once
                # Forward pass through the transformer
                head_inputs_dict = self.forward_transformer(copy.deepcopy(visual_feats), copy.deepcopy(text_dict), batch_data_samples)
                # Predict instances
                pred_instances = self.bbox_head.predict(**head_inputs_dict,
                                                        rescale=rescale,
                                                        batch_data_samples=batch_data_samples)[0]
                # If there are predicted instances, update their labels
                if len(pred_instances) > 0:
                    pred_instances.labels += count
                # Update the label counter
                count += len(token_positive_maps_once)
                # Append the predicted instances to the original results list
                origin_results_list.append(pred_instances)
            # Concatenate prediction results
            origin_results_list = origin_results_list[0].cat(origin_results_list)
            # If there are prediction results, create a new results list
            if len(origin_results_list) > 0:
                results_list = [origin_results_list]
                is_rec_tasks = [False] * len(results_list)

        else:
            # Extract text features
            text_dict = self.language_model(list(text_prompts))
            # Text feature mapping layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            # Initialize a list to mark whether each task is an entity recognition task
            is_rec_tasks = []
            # Mark whether it is an entity recognition task
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                # Assign the token positive map to the data sample
                data_samples.token_positive_map = token_positive_maps[i]

            # Forward pass through the transformer
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            # Predict instances
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # Assign predicted instances and label names to each data sample
        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            # If there are predicted instances
            if len(pred_instances) > 0:
                # Initialize a list to store label names
                label_names = []
                # Iterate through the labels of the predicted instances
                for labels in pred_instances.labels:
                    # If it is an entity recognition task
                    if is_rec_task:
                        # Append the entity to the label names list
                        label_names.append(entity)
                        continue
                    # If the label index is out of bounds
                    if labels >= len(entity):
                        # Issue a warning
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        # Append 'unobject' to the label names list
                        label_names.append('unobject')
                    else:
                        # Append the corresponding entity to the label names list
                        label_names.append(entity[labels])
                # Assign the label names to the predicted instances for visualization
                pred_instances.label_names = label_names
            # Assign the predicted instances to the data sample
            data_sample.pred_instances = pred_instances
        # Return the batch of data samples with predicted instance information
        return batch_data_samples