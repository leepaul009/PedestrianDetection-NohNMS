# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
import math
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, allow_oob=False
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            allow_oob,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, allow_oob=False
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    if not allow_oob:
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    else:
        boxes = boxes.view(-1, num_bbox_reg_classes, 4)

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    from torchvision.ops import nms

    keep = nms(boxes, scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
        giou=False,
        allow_oob=False,
        sample_type=None,
        loss_weight_box=1.0,
        loss_weight_logic=1.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.giou = giou
        self.allow_oob = allow_oob
        self.sample_type = sample_type
        self.loss_weight_box = loss_weight_box
        self.loss_weight_logic = loss_weight_logic

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        
        self.loss_per_image = None

        if False:
            self.proposal_size_per_image = None
            if proposals[0].has("gt_boxes"):
                # self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                # assert proposals[0].has("gt_classes")
                # self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                
                self.proposal_size_per_image = [ p.gt_classes.shape[0] for p in proposals ]
                pos1 = 0
                pos2 = 0
                self.loss_per_image = torch.zeros([len(proposals)], dtype=torch.float)
                for i, p in enumerate(proposals):
                    pos2 += p.gt_classes.shape[0]
                    self.loss_per_image[i] = F.cross_entropy(self.pred_class_logits[pos1 : pos2], 
                                                            p.gt_classes, 
                                                            reduction="mean").detach()
                    pos1 = pos2


    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def matching_softmax_cross_entropy_loss(self):
        max_score, _ = torch.max(self.pred_class_logits[:, :-1], dim=1)
        pred_class_logits = torch.stack([max_score, self.pred_class_logits[:, -1]], axis=1)
        return F.cross_entropy(pred_class_logits, self.gt_classes, reduction="mean")

    def matching_softmax_cross_entropy_loss1(self):

        max_score, _ = torch.max(self.pred_class_logits[:, :-1], dim=1)
        pred_class_logits = torch.stack([max_score, self.pred_class_logits[:, -1]], axis=1)
        pos_loss = F.cross_entropy(pred_class_logits, self.gt_classes, reduction="none")

        max_score, max_indice = torch.max(self.pred_class_logits[:, :-1], dim=1)
        gt_classes = max_indice * (1 - self.gt_classes) + self.gt_classes * (
            self.pred_class_logits.shape[1] - 1
        )
        neg_loss = F.cross_entropy(self.pred_class_logits, gt_classes, reduction="none")

        total_loss = pos_loss * (1 - self.gt_classes) + neg_loss * self.gt_classes

        return torch.mean(total_loss)

    def matching_softmax_cross_entropy_loss2(self, alpha=1.0, beta=0.3):

        min_score, _ = torch.min(self.pred_class_logits[:, :-1], dim=1)
        pred_class_logits = torch.stack([min_score, self.pred_class_logits[:, -1]], axis=1)
        pos_loss2 = F.cross_entropy(pred_class_logits, self.gt_classes, reduction="none")

        _, max_indice = torch.max(self.pred_class_logits[:, :-1], dim=1)
        gt_classes = max_indice * (1 - self.gt_classes) + self.gt_classes * (
            self.pred_class_logits.shape[1] - 1
        )
        loss = F.cross_entropy(self.pred_class_logits, gt_classes, reduction="none")

        total_loss = alpha * loss + beta * pos_loss2 * (1 - self.gt_classes)

        return torch.mean(total_loss)

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        # VERY IMPORTANT!!!!!!!!
        # should = num_class
        # bg_class_ind = 1  # self.pred_class_logits.shape[1] - 1
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # print( "pred_proposal_deltas: {}, {}".format( self.pred_proposal_deltas.size(),
        #                                              self.pred_proposal_deltas ) )

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        
        # print( "gt_classes: {}".format( self.gt_classes ) )

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )

        
        #################
        if True and self.sample_type == "RepeatFactorTrainingSampler":
            max_score, _      = torch.max(self.pred_class_logits[:, :-1], dim=1)
            pred_class_logits = torch.stack([max_score, self.pred_class_logits[:, -1]], axis=1)
            
            #################
            # get valid prediction indexes
            # self.gt_classes is label of predictions

            valid_inds = torch.nonzero( (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind) ).squeeze(1).detach().cpu()

            # print("######## valid_inds_all_img: {}".format( valid_inds ))

            # get split per image
            # self.num_preds_per_image is a list, each item is number of predictions per image

            self.loss_per_image = torch.zeros( [len(self.num_preds_per_image)], dtype=torch.float )
            pos_beg = 0
            pos_end = 0
            for i, num_preds in enumerate(self.num_preds_per_image):
                pos_end += num_preds 

                # find split of valid prediction indexes within [pos_beg, pos_end)

                selected_inds_to_valid_inds = torch.nonzero( (valid_inds >= pos_beg) & (valid_inds < pos_end) ).squeeze(1)

                self.loss_per_image[i] += F.cross_entropy(pred_class_logits[pos_beg:pos_end], 
                                                        self.gt_classes[pos_beg:pos_end], 
                                                        reduction="mean").detach().cpu()
                
                # print("######## image {}: {} -> {}".format( i, pos_beg, pos_end ))
                # print("######## image {}: cls_loss {}".format( i, self.loss_per_image[i] ))

                pos_beg += num_preds

                num_selected_box = selected_inds_to_valid_inds.shape[0]
                if num_selected_box == 0:
                    continue

                valid_inds_this_img = valid_inds[selected_inds_to_valid_inds]

                # print("######## image {}: valid_inds_this_img: {}".format( i, valid_inds_this_img ))

                if cls_agnostic_bbox_reg:
                    gt_class_cols = torch.arange(box_dim, device=device)
                else:
                    fg_gt_classes = self.gt_classes[valid_inds_this_img]
                    gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

                self.loss_per_image[i] += smooth_l1_loss(self.pred_proposal_deltas[valid_inds_this_img[:,None], gt_class_cols],
                                                        gt_proposal_deltas[valid_inds_this_img], 
                                                        self.smooth_l1_beta,
                                                        reduction="sum").detach().cpu() / num_selected_box
                
                # print("######## image {}: reg_loss {}".format( i, self.loss_per_image[i] ))

        
        # print( "fg_inds[]: {}".format( fg_inds[:, None] ) )
        # print( "gt_class_cols: {}".format( gt_class_cols ) )
        # print( "pred_proposal_deltas[]: {}".format( 
        #     self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] ) )

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.

        # loss_box_reg = loss_box_reg / self.gt_classes.numel()
        loss_box_reg = loss_box_reg / self.pred_proposal_deltas.shape[0]
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
            self.giou: def false
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.matching_softmax_cross_entropy_loss()*self.loss_weight_logic,
            "loss_box_reg": self.smooth_l1_loss()*self.loss_weight_box \
                            if not self.giou else self.giou_loss()*self.loss_weight_box,
        }

    def giou_loss(self, eps=1e-7):
        """
        Generalized Intersection over Union: A Metric and A Loss for
        Bounding Box Regression
        https://arxiv.org/abs/1902.09630

        code refer to:
        https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).

        Return:
            Tensor: Loss tensor.
        """
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )

        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        target = self.gt_boxes.tensor

        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

        # GIoU
        gious = ious - (enclose_area - union) / enclose_area
        loss = 10 * (1 - gious[fg_inds]).sum()
        loss = loss / self.gt_classes.numel()
        return loss

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def predict_multi_probs(self):
        max_fg_prob, _ = torch.max(self.pred_class_logits[:, :-1], dim=1)
        probs = torch.stack([max_fg_prob, self.pred_class_logits[:, -1]], axis=1)
        probs = F.softmax(probs, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
            allow_oob=self.allow_oob,
        )


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

# new layer
class FastRCNN2TransformerLayers(nn.Module):
    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNN2TransformerLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        # self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        self.num_encoder = 4
        self.hidden_dim = 256

        self.src_embedding = nn.Linear(input_size, self.hidden_dim)
        self.cls_score = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(self.hidden_dim, num_bbox_reg_classes * box_dim)
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, nhead=8, 
            num_encoder_layers=self.num_encoder, 
            num_decoder_layers=self.num_encoder)
        # sequence length = max number of proposals per batch
        # VERY IMPORTANT: sequence length should be longer than max number of proposals per img
        self.seq_len = 1024
        self.query_pos = nn.Parameter(torch.rand(self.seq_len, self.hidden_dim))


        nn.init.normal_(self.src_embedding.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.src_embedding]:
            nn.init.constant_(l.bias, 0)

    # x: [n_propsals, 1024]
    def forward(self, x, proposals):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        device = x.device

        # list, num_proposals of each batch
        # during inference, num_instance in proposal_per_image is 1000
        # proposal_deltas = self.bbox_pred(x) # [P, 1024] => [P, 4]

        x = self.src_embedding(x) # [P, 1024] => [P, 256]
        num_preds_per_image = [len(p) for p in proposals]
        x = x.split(num_preds_per_image, dim=0)

        enc_x = []
        for i, n_preds in enumerate(num_preds_per_image):
            # tgt_mask = torch.zeros(self.seq_len, self.seq_len)
            tgt_key_padding_mask = torch.ones(1, self.seq_len, dtype=torch.bool, device=device) # [N=1, 1024]
            tgt_key_padding_mask[ :, 0:n_preds ] = False # [N=1, P] <= not mask
            
            y = self.transformer(x[i].unsqueeze(1),          # [P, 256] => [P, N=1, 256]
                                self.query_pos.unsqueeze(1), # [1024, 256]] => [1024, N=1, 256]]
                                # tgt_mask=tgt_mask, 
                                tgt_key_padding_mask = tgt_key_padding_mask, # [N=1, 1024]
                                )
            enc_x.append( y.squeeze(1)[0:n_preds, :] ) # [T, N=1, H] => [T, H] => [T', H]
        enc_x = torch.cat(enc_x, dim=0)

        scores          = self.cls_score(enc_x) # [P, 256] => [P, 2]
        proposal_deltas = self.bbox_pred(enc_x) # [P, 256] => [P, 4]

        return scores, proposal_deltas

class FastRCNN2TransformerLayers1(nn.Module):
    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNN2TransformerLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        self.num_encoder = 4
        self.hidden_dim = 256

        self.src_embedding = nn.Linear(input_size, self.hidden_dim)
        self.cls_score = nn.Linear(self.hidden_dim, num_classes + 1)
        # self.bbox_pred = nn.Linear(self.hidden_dim, num_bbox_reg_classes * box_dim)
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, nhead=8, 
            num_encoder_layers=self.num_encoder, 
            num_decoder_layers=self.num_encoder)
        # sequence length = max number of proposals per batch
        # VERY IMPORTANT: sequence length should be longer than max number of proposals per img
        self.seq_len = 1024
        self.query_pos = nn.Parameter(torch.rand(self.seq_len, self.hidden_dim))


        nn.init.normal_(self.src_embedding.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.src_embedding]:
            nn.init.constant_(l.bias, 0)

    # x: [n_propsals, 1024]
    def forward(self, x, proposals):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        device = x.device

        # list, num_proposals of each batch
        # during inference, num_instance in proposal_per_image is 1000
        proposal_deltas = self.bbox_pred(x) # [P, 1024] => [P, 4]

        x = self.src_embedding(x) # [P, 1024] => [P, 256]
        num_preds_per_image = [len(p) for p in proposals]
        x = x.split(num_preds_per_image, dim=0)

        enc_x = []
        for i, n_preds in enumerate(num_preds_per_image):
            # tgt_mask = torch.zeros(self.seq_len, self.seq_len)
            tgt_key_padding_mask = torch.ones(1, self.seq_len, dtype=torch.bool, device=device) # [N=1, 1024]
            tgt_key_padding_mask[ :, 0:n_preds ] = False # [N=1, P] <= not mask
            
            y = self.transformer(x[i].unsqueeze(1),          # [P, 256] => [P, N=1, 256]
                                self.query_pos.unsqueeze(1), # [1024, 256]] => [1024, N=1, 256]]
                                # tgt_mask=tgt_mask, 
                                tgt_key_padding_mask = tgt_key_padding_mask, # [N=1, 1024]
                                )
            enc_x.append( y.squeeze(1)[0:n_preds, :] ) # [T, N=1, H] => [T, H] => [T', H]
        enc_x = torch.cat(enc_x, dim=0)

        scores          = self.cls_score(enc_x) # [P, 256] => [P, 2]
        # proposal_deltas = self.bbox_pred(enc_x) # [P, 256] => [P, 4]

        return scores, proposal_deltas



class AttRCNNLayers(nn.Module):

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, is_train=True):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(AttRCNNLayers, self).__init__()

        self.is_train = is_train
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        ### input_size(1024) -> 2/4, when C=1
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        # num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        # self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)


        ### attention
        self.num_encoder = 4
        self.hidden_dim = 256

        ### transformer -> out
        ### [proposals, 256] -> [proposals, 2/4], when C=1
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        # self.bbox_pred = nn.Linear(self.hidden_dim, num_bbox_reg_classes * box_dim)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.cls_score = nn.Linear(self.hidden_dim, num_classes + 1)
        


        ### input and target embedding
        ### [proposals, 1024] => [proposals, 256]
        self.src_embedding = nn.Linear(input_size, self.hidden_dim)
        ### [proposals, 5(cls+box)] => [proposals, 256]
        self.tgt_embedding = nn.Linear(5, self.hidden_dim)

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, nhead=8, 
            num_encoder_layers=self.num_encoder, 
            num_decoder_layers=self.num_encoder)
        ### seq_len: max number of proposals per image
        ### VERY IMPORTANT: seq_len >=  max number of proposals per image
        # self.seq_len = 1024
        # self.query_pos = nn.Parameter(torch.rand(self.seq_len, self.hidden_dim))
        # self.predictor = nn.Linear(self.hidden_dim, num_classes + 1) # [proposals, 256] => [proposals, 2]
        
        nn.init.normal_(self.src_embedding.weight, std=0.01)
        nn.init.normal_(self.tgt_embedding.weight, std=0.01)
        # nn.init.normal_(self.predictor.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.src_embedding, self.tgt_embedding]:
            nn.init.constant_(l.bias, 0)

    # x: [n_propsals, 1024]
    def forward(self, x, proposals):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        device = x.device

        # list, num_proposals of each batch
        # during inference, num_instance in proposal_per_image is 1000
        # define self.bbox_pred as (1024->4)
        proposal_deltas = self.bbox_pred(x) # [propsals, 1024] =>  [propsals, 4]

        # is_train = False
        # if proposals[0].has("gt_boxes"):
        #     is_train = True

        x = self.src_embedding(x) # [proposals, 1024] => [proposals, 256]
        
        num_preds_per_image = [len(p) for p in proposals]
        x = x.split(num_preds_per_image, dim=0) # => list of x_per_image, [x0, x1 ...]
        outs = []
        ### for each image
        for i in range(len(num_preds_per_image)):
            
            if proposals[0].has("gt_boxes"): # is train
                gt_box = proposals[i].gt_boxes.tensor # [proposals, 4]
                gt_cls = proposals[i].gt_classes      # [proposals,]
                tgt_emb = torch.cat([gt_box, gt_cls.unsqueeze(1)], dim=1)
                tgt_emb = self.tgt_embedding(tgt_emb).unsqueeze(1) # [proposals, 5] => [proposals, 256] => [proposals, 1, 256]
                out = self.transformer(
                        x[i].unsqueeze(1), # [T, N=1, H]
                        tgt_emb,           # [T, N=1, H]
                        )
                outs.append( out.squeeze(1) ) # [T, N=1, H] => [T, H]
            else:
                # self.transformer.encoder(src, src_mask)
                # self.transformer.decoder(tgt, memory, gt_mask)
                memory = self.transformer.encoder(x[i].unsqueeze(1))

                ########## PED TODO:
                tgt = torch.rand(num_preds_per_image[i], 5)
                tgt_emb = self.tgt_embedding(tgt).unsqueeze(1)
                out = self.transformer.decoder(tgt_emb, memory)

                '''
                output = torch.ones(1, 5)
                for t in range(1, num_preds_per_image[i]):
                    tgt_emb = self.tgt_embedding(output[:, :t]).unsqueeze(1) # [T, 5]=>[T, 1, H]
                    decoder_output  = self.transformer.decoder(tgt_emb, memory)
                    pred_proba_t = self.predictor(decoder_output) # [T, 1, H] => [T, 1, 2]
                    output_t = pred_proba_t.max(-1)[1] # [T, 1, 2] => [T, 1]
                    output[:, t] = output_t
                '''

                outs.append( out.squeeze(1) ) # [T, N=1, H] => [T, H]
        outs = torch.cat(outs, dim=0)
        scores = self.cls_score(outs) # [propsals, 256] => [propsals, 2]
        return scores, proposal_deltas
