from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import numpy as np
import pycocotools.mask as mask_util
import torch.nn.functional as F
import time

@DETECTORS.register_module
class Solo(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Solo, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, 
                             img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_list=[bbox_list]
        img_meta = None if not self.test_cfg.get('cpu_test', False) else img_meta[0]['cpu_postprocess']
        bbox_results = [
            bbox_mask2result(det_bboxes, det_labels, det_masks, self.bbox_head.num_classes,
                             img_meta=img_meta)
            for det_bboxes, det_labels, det_masks in bbox_list
        ]
        return bbox_results[0][0],bbox_results[0][1]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


def bbox_mask2result(bboxes,labels, masks, num_classes, img_meta=None):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n,)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """

    if img_meta is not None:
        (pad_h, pad_w) = img_meta['pad_shape']
        (crop_h, crop_w) = img_meta['img_shape']
        ori_shape = img_meta['ori_shape']
        mask_thr = img_meta['mask_thr']
        masks = masks.cpu()
        if img_meta.get('approx', True):
            mask_h, mask_w = masks.shape[-2:]
            valid_h, valid_w = int((crop_h / pad_h) * mask_h), int((crop_w / pad_w) * mask_w)
            mask_preds = masks[:, :valid_h, :valid_w]
            mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=True)[0]
        else:
            mask_preds = F.upsample_bilinear(masks.unsqueeze(0), (pad_h, pad_w))
            mask_preds = mask_preds[:, :, :crop_h, :crop_w]
            mask_preds = F.upsample_bilinear(mask_preds, ori_shape)[0]
        masks = mask_preds > mask_thr

    mask_results = [[] for _ in range(num_classes - 1)]
    for i in range(masks.shape[0]):
        im_mask=masks[i].int().data.cpu().numpy().astype(np.uint8)
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]

        label = labels[i]

        mask_results[label].append(rle)
    
    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results
    else:
        bboxes = bboxes
        labels = labels
        bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        return bbox_results, mask_results
