# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher, MatcherIgnore
from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .overlap_head import OverlapFastRCNNOutputs, OverlapOutputLayers

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class CascadeMutationROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        cascade_iou_label        = cfg.MODEL.ROI_HEADS.IOU_LABELS # [0, 1]

        ## OVERLAP
        self.giou                = cfg.MODEL.ROI_BOX_HEAD.GIoU
        self.allow_oob           = cfg.MODEL.ALLOW_BOX_OUT_OF_BOUNDARY
        self.overlap_enable      = cfg.MODEL.OVERLAP_BOX_HEAD.ENABLE

        self.num_cascade_stages  = len(cascade_ious)
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        self.box_head = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        self.box2box_transform = []
        self.proposal_matchers = []
        for k in range(self.num_cascade_stages):
            box_head = build_box_head(cfg, pooled_shape)
            self.box_head.append(box_head)
            self.box_predictor.append(
                FastRCNNOutputLayers(
                    box_head.output_size, self.num_classes, cls_agnostic_bbox_reg=True
                )
            )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            ############
            ## OVERLAP: when last stage
            if self.overlap_enable and k == self.num_cascade_stages-1: 
                # self._init_overlap_head(cfg, input_shape, in_channels, pooler_resolution)
                self.build_on_roi_feature = cfg.MODEL.OVERLAP_BOX_HEAD.BUILD_ON_ROI_FEATURE
                self.sigmoid_on           = cfg.MODEL.OVERLAP_BOX_HEAD.SIGMOID_ON
                self.overlap_configs = {
                    "overlap_iou_threshold":  cfg.MODEL.OVERLAP_BOX_HEAD.OVERLAP_IOU_THRESHOLD,
                    "loss_overlap_reg_coeff": cfg.MODEL.OVERLAP_BOX_HEAD.REG_LOSS_COEFF,
                    "uniform_reg_divisor":    cfg.MODEL.OVERLAP_BOX_HEAD.UNIFORM_REG_DIVISOR,
                    "cls_box_beta":           cfg.MODEL.OVERLAP_BOX_HEAD.PROB_LOSS_BETA,
                }
                if self.build_on_roi_feature:
                    self.overlap_head = build_box_head(
                        cfg,
                        ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution),
                    )
                self.overlap_predictor = OverlapOutputLayers(
                    box_head.output_size, 
                    num_classes = self.num_classes, 
                    sigmoid_on  = self.sigmoid_on
                )
            ############

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    # Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                    # cascade_iou_label [0, 1]
                    MatcherIgnore( [cascade_ious[k]], cascade_iou_label )
                )

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets, self.overlap_enable)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.in_features]
        head_outputs = []
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                proposals = self._create_proposals_from_boxes(
                    head_outputs[-1].predict_boxes(), image_sizes
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            if self.overlap_enable and k == self.num_cascade_stages - 1:
                head_outputs.append(self._run_stage_overlap(features, proposals, k))
            else:
                head_outputs.append(self._run_stage(features, proposals, k))

        if self.training:
            losses = {}
            storage = get_event_storage()
            self.loss_per_image = None
            for stage, output in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = output.losses()
                    if output.loss_per_image != None:
                        if self.loss_per_image == None:
                            self.loss_per_image = output.loss_per_image
                        else:
                            self.loss_per_image += output.loss_per_image
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]

            # Average the scores across heads, [num_pred, num_classes+1]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            boxes = head_outputs[-1].predict_boxes()
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            match_quality_matrix_t = match_quality_matrix.transpose(1, 0)
            ignore_match_quality_matrix_t = pairwise_ioa(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes).transpose(1, 0)
            
            gt_ignore_mask = targets_per_image.gt_classes.eq(-1).repeat(
                ignore_match_quality_matrix_t.shape[0], 1
            ) # [pred, gt]
            match_quality_matrix_t        *= ~gt_ignore_mask # remove ignored gt
            ignore_match_quality_matrix_t *= gt_ignore_mask  # remove valid gt
            
            # proposal_labels: 0, 1, -1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](
                match_quality_matrix_t,
                ignore_match_quality_matrix_t,
                targets_per_image.gt_classes,
            )
            
            # proposal_labels are 0 or 1
            ## matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)

            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                ## gt_classes[proposal_labels == 0] = self.num_classes
                gt_classes[proposal_labels <= 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes


            if self.overlap_enable:
                matched_vals, sorted_idx = match_quality_matrix.sort(0, descending=True)
                if matched_vals.size(0) > 1:
                    # assign second large IoU
                    overlap_iou    = matched_vals[1, :]
                    overlap_gt_idx = sorted_idx[1, :]
                else:
                    overlap_iou    = matched_vals.new_zeros(matched_vals.size(1))
                    overlap_gt_idx = sorted_idx[0, :]
                selected_overlap_iou      = overlap_iou
                selected_overlap_gt_idx   = overlap_gt_idx
                selected_overlap_gt_boxes = targets_per_image.gt_boxes[selected_overlap_gt_idx]
                proposals_per_image.overlap_iou      = selected_overlap_iou
                proposals_per_image.overlap_gt_boxes = selected_overlap_gt_boxes


            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            FastRCNNOutputs: the output of this stage
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform[stage],
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            giou=self.giou,
            allow_oob=self.allow_oob,
            sample_type=self.sample_type,
            loss_weight_box=self.loss_weight_box,
            loss_weight_logic=self.loss_weight_logic,
        )
        return outputs

    def _run_stage_overlap(self, features, proposals, stage):

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        if self.build_on_roi_feature:
            overlap_features = self.overlap_head(box_features)
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)
        pred_overlap_prob, pred_overlap_deltas  = self.overlap_predictor(overlap_features)
        del box_features
        del overlap_features

        # outputs = FastRCNNOutputs(
        #     self.box2box_transform[stage],
        #     pred_class_logits,
        #     pred_proposal_deltas,
        #     proposals,
        #     self.smooth_l1_beta,
        # )
        outputs = OverlapFastRCNNOutputs(
                self.box2box_transform[stage],
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                pred_overlap_deltas=pred_overlap_deltas,
                pred_overlap_prob=pred_overlap_prob,
                overlap_configs=self.overlap_configs,
                giou=self.giou,
                allow_oob=self.allow_oob,
                sample_type=self.sample_type, ## from roi_head
                loss_weight_box=self.loss_weight_box, ## from roi_head
                loss_weight_logic=self.loss_weight_logic, ## from roi_head
            )

        return outputs

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
