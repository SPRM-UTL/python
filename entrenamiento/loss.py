import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from aplicacion.boxes import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, weight_dict: dict):
        super().__init__()
        self.class_weighting = weight_dict["class_weighting"]
        self.bbox_weighting = weight_dict["bbox_weighting"]
        self.giou_weighting = weight_dict["giou_weighting"]

    @torch.no_grad()
    def forward(self, yhat, y):
        indices = []

        for batch_idx, target in enumerate(y):
            batch_logits = yhat["pred_logits"][batch_idx]
            batch_boxes = yhat["pred_boxes"][batch_idx]
            batch_prob = batch_logits.softmax(-1)

            target_labels = target["labels"].to(torch.long)
            target_boxes = target["boxes"].to(batch_boxes.dtype)

            cost_class = -batch_prob[:, target_labels]
            cost_bbox = torch.cdist(batch_boxes, target_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(batch_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )

            cost_matrix = (
                self.bbox_weighting * cost_bbox
                + self.class_weighting * cost_class
                + self.giou_weighting * cost_giou
            ).cpu()

            prediction_indices, target_indices = linear_sum_assignment(cost_matrix)
            indices.append(
                (
                    torch.as_tensor(prediction_indices, dtype=torch.int64),
                    torch.as_tensor(target_indices, dtype=torch.int64),
                )
            )

        return indices


class DETRLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, yhat, y):
        y = [
            {
                "labels": target["labels"].to(torch.long),
                "boxes": target["boxes"].to(torch.float32),
            }
            for target in y
        ]
        indices = self.matcher(yhat, y)

        device = next(iter(yhat.values())).device
        num_boxes = sum(len(target["labels"]) for target in y)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device).clamp(min=1)

        return {
            "labels": self.classification_loss(yhat, y, indices),
            "boxes": self.box_loss(yhat, y, indices, num_boxes),
        }

    def classification_loss(self, yhat, y, indices):
        src_logits = yhat["pred_logits"]
        idx = self.get_matched_query_indices(indices)

        target_classes_o = torch.cat([target["labels"][j] for target, (_, j) in zip(y, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        return {
            "loss_ce": F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight,
            )
        }

    def box_loss(self, yhat, y, indices, num_boxes):
        idx = self.get_matched_query_indices(indices)
        src_boxes = yhat["pred_boxes"][idx]
        target_boxes = torch.cat([target["boxes"][i] for target, (_, i) in zip(y, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )

        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def get_matched_query_indices(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
