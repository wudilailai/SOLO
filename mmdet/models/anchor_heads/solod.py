import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms_with_mask
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import numpy as np
import itertools

INF = 1e8
import time

@HEADS.register_module
class DecoupledSoloHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 radius=0.2,
                 dice_weight=3.0,
                 grid_num=[40, 36, 24, 16, 12],
                 strides=(8, 8, 16, 32, 32),
                 use_mass_center=False,
                 mask_upsample=False,
                 regress_ranges=((-1, 96), (48, 192), (96, 384), (192, 768),
                                 (384, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(DecoupledSoloHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.regress_ranges = regress_ranges
        self.use_mass_center = use_mass_center
        self.mask_upsample = mask_upsample
        self.strides = strides
        self.loss_cls = build_loss(loss_cls)
        # self.loss_mask = build_loss(loss_mask)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.grid_num = grid_num

        self.radius = radius
        self.dice_weight = dice_weight
        self._init_layers()

        if self.mask_upsample:
            self.strides = [s // 2 for s in strides]

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            if i == 0:
                self.mask_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=dict(type='CoordConv'),
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
            else:
                self.mask_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
        self.solo_cls = nn.ModuleList([nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 1, padding=0) for _ in self.grid_num])
        # use decouple head
        self.solo_mask_x = nn.ModuleList([nn.Conv2d(
            self.feat_channels, num, 1, padding=0) for num in self.grid_num])
        self.solo_mask_y = nn.ModuleList([nn.Conv2d(
            self.feat_channels, num, 1, padding=0) for num in self.grid_num])
        '''
        self.solo_mask = nn.ModuleList([nn.Conv2d(
            self.feat_channels, num ** 2, 1, padding=0) for num in self.grid_num])
        '''
        ori_strides = [4, 8, 16, 32, 64]
        self.stride_scales = [ori / new for ori, new in zip(ori_strides, self.strides)]

    def init_weights(self):
        # ConvModule 已经有init了
        '''
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        '''
        bias_cls = bias_init_with_prob(0.01)
        for m in self.solo_cls:
            normal_init(m, std=0.01, bias=bias_cls)
        for m in self.solo_mask_x:
            normal_init(m, std=0.01)
        for m in self.solo_mask_y:
            normal_init(m, std=0.01)

    def forward(self, feats):
        cls_score, mask_score_x, mask_score_y = multi_apply(self.forward_single, feats, self.solo_cls, self.solo_mask_x, self.solo_mask_y, self.grid_num,
                                            self.stride_scales)
        return cls_score, mask_score_x, mask_score_y

    def forward_single(self, x, solo_cls, solo_mask_x, solo_mask_y, grid_num, stride_scale):
        if stride_scale != 1:
            x = F.interpolate(x, scale_factor=stride_scale, mode="bilinear")

        cls_feat = F.upsample_bilinear(x, (grid_num, grid_num))
        mask_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = solo_cls(cls_feat)

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        if self.mask_upsample:
            mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear', align_corners=True)
        mask_score_x = solo_mask_x(mask_feat)
        mask_score_y = solo_mask_y(mask_feat)
        return cls_score, mask_score_x, mask_score_y

    def dice_loss(self,input, target):
        '''
        print(input.shape, target.shape)
        smooth = 0.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))
        '''
        num = target.size(0)
        smooth = 1
        if num == 0:
            return torch.tensor(1.0)
        m1 = input.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / ((m1*m1).sum(1) + (m2*m2).sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

    def get_points(self, featmap_sizes, strides, dtype, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride[0], stride[0], dtype=dtype, device=device) + stride[0] // 2
        y_range = torch.arange(
            0, h * stride[1], stride[1], dtype=dtype, device=device) + stride[1] // 2
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1)
        return points

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1, gt_centers=None):
        if not self.use_mass_center:
            center_x = (gt[..., 0] + gt[..., 2]) / 2
            center_y = (gt[..., 1] + gt[..., 3]) / 2
        else:
            gt_centers = gt_centers.expand(*gt.shape[:2], 2)
            center_x, center_y = gt_centers[..., 0], gt_centers[..., 1]
        w = gt[..., 2] - gt[..., 0]
        h = gt[..., 3] - gt[..., 1]
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0

        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            xmin = center_x[beg:end] - w[beg:end] * radius
            ymin = center_y[beg:end] - h[beg:end] * radius
            xmax = center_x[beg:end] + w[beg:end] * radius
            ymax = center_y[beg:end] + h[beg:end] * radius
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def solo_target(self, points, gt_bboxes_list, gt_labels_list, strides, gt_centers_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # split to per img, per level
        num_points = [center.size(0) for center in points]
        self.num_points_per_level = num_points
        # get labels and bbox_targets of each image
        labels_list, inds_list = multi_apply(
            self.solo_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_centers_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            cls_strides=strides)
        labels_list = [labels.split(num_points, 0) for labels in labels_list]

        inds_list = [inds.split(num_points, 0) for inds in inds_list]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_inds = []
        num_imgs = len(labels_list)
        inds_img = []
        for j in range(5):
            for i in range(num_imgs):
                size = len(labels_list[i][j])
                inds_img.append([i for k in range(size)])
        inds_img = list(itertools.chain.from_iterable(inds_img))

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_inds.append(
                torch.cat(
                    [inds[i] for inds in inds_list]))
        return concat_lvl_labels, concat_lvl_inds, inds_img

    def solo_target_single(self, gt_bboxes, gt_labels, gt_centers, points, regress_ranges, cls_strides):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # keep alignment !!!

        inside_gt_bbox_mask = self.get_sample_region(gt_bboxes,
                                                     cls_strides,
                                                     self.num_points_per_level,
                                                     xs,
                                                     ys,
                                                     radius=self.radius,
                                                     gt_centers=gt_centers)

        # condition2: limit the regression range for each location
        regress_ranges = regress_ranges * regress_ranges
        inside_regress_range = (areas >= regress_ranges[..., 0]) & (areas <= regress_ranges[..., 1])
        # max_regress_distance = bbox_targets.max(-1)[0]
        # inside_regress_range = (
        #                                max_regress_distance >= regress_ranges[..., 0]) & (
        #                                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0

        return labels, min_area_inds

    def get_mass_centers(self, gt_masks_list):
        if self.use_mass_center:
            device = 'cpu'
            centers_list = []
            for gt_masks_per_img in gt_masks_list:
                h, w = gt_masks_per_img.shape[-2:]
                y = torch.linspace(0, h-1, h, device=device)
                x = torch.linspace(0, w-1, w, device=device)
                ys, xs = torch.meshgrid([y, x])
                masks = torch.from_numpy(gt_masks_per_img).to(device).float()
                n_mask = masks.size(0)
                n_fg = masks.sum((1, 2))
                y_centers = (ys.expand(n_mask, h, w) * masks).sum((1, 2)) / n_fg
                x_centers = (xs.expand(n_mask, h, w) * masks).sum((1, 2)) / n_fg
                centers_list.append(torch.stack([x_centers, y_centers], dim=1).to('cuda'))
        else:
            centers_list = [None for _ in gt_masks_list]
        return centers_list

    def loss(self,
             cls_scores,
             mask_preds_x,
             mask_preds_y,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(mask_preds_x) == len(mask_preds_y)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # featmap_sizes == grid num
        mask_sizes = [mask.size()[-2:] for mask in mask_preds_x]
        cls_strides = []
        level = len(self.strides)
        for i in range(level):
            f_y, f_x = featmap_sizes[i]
            m_y, m_x = mask_sizes[i]
            s_y = m_y / f_y * self.strides[i]
            s_x = m_x / f_x * self.strides[i]
            cls_strides.append([s_x, s_y])

        # get mass centers
        centers_list = self.get_mass_centers(gt_masks)

        all_level_points = self.get_points(featmap_sizes, cls_strides, cls_scores[0].dtype,
                                           cls_scores[0].device)
        labels, inds, img_inds = self.solo_target(all_level_points, gt_bboxes,
                                                  gt_labels, cls_strides, gt_centers_list=centers_list)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in
                              cls_scores]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels = torch.cat(labels)
        flatten_inds = torch.cat(inds)
        img_inds = torch.tensor(img_inds).to(flatten_inds.device)
        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels,
                                 avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        # flatten mask pred
        mask_preds_list = []
        mask_targets_list = []
        ins_num = 0
        for i in range(5):
            _, _, b_h, b_w = mask_preds_x[0].shape
            grid_num = self.grid_num[i]
            pos_inds = labels[i].nonzero().reshape(-1)
            pos_ins_y = pos_inds//grid_num
            pos_ins_x = pos_inds%grid_num

            pos_ins_inds = inds[i][pos_inds]

            mask_preds_x[i] = F.sigmoid(F.upsample_bilinear(mask_preds[i], (b_h, b_w)).view(-1, b_h, b_w)[pos_ins_x])
            mask_preds_y[i] = F.sigmoid(F.upsample_bilinear(mask_preds[i], (b_h, b_w)).view(-1, b_h, b_w)[pos_ins_y])
            #import IPython 
            #IPython.embed()
            pos_img_inds = img_inds[pos_inds + ins_num]
            ins_num += len(labels[i])
            mask_targets = torch.zeros((len(pos_inds), b_h, b_w)).to(mask_preds_x[0].device)
            for j in range(num_imgs):
                _, i_h, i_w = gt_masks[j].shape
                temp_mask = nn.ConstantPad2d((0, b_w * self.strides[i] - i_w, 0, b_h * self.strides[i] - i_h), 0)(
                    torch.tensor(gt_masks[j])).to(mask_preds_x[0].device)
                temp_mask = F.upsample_bilinear(temp_mask.float().unsqueeze(0), (b_h, b_w))[0]
                ind_this_img = torch.nonzero(pos_img_inds == j).flatten()
                ins_this_img = pos_ins_inds[ind_this_img]
                mask_targets[ind_this_img] = temp_mask[ins_this_img]

            mask_preds_list.append(mask_preds_x[i].mul(mask_preds_y[i]))
            mask_targets_list.append(mask_targets)
        mask_preds = torch.cat(mask_preds_list)
        mask_targets = torch.cat(mask_targets_list)
        loss_mask = self.dice_loss(mask_preds, mask_targets) * self.dice_weight
        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask)
        '''
        _, _, b_h, b_w = mask_preds_x[0].shape
        for i in range(5):
            mask_preds_x[i] = F.sigmoid(F.upsample_bilinear(mask_preds_x[i], (b_h, b_w))).view(-1, b_h, b_w)
            mask_preds_y[i] = F.sigmoid(F.upsample_bilinear(mask_preds_y[i], (b_h, b_w))).view(-1, b_h, b_w)

        mask_preds = torch.cat([mask_pred for mask_pred in mask_preds], dim=0)

        #decouple
        mask_preds = mask_preds[pos_inds]
        pos_ins_inds = flatten_inds[pos_inds]
        pos_img_inds = img_inds[pos_inds]
        mask_target = torch.zeros((len(pos_inds), b_h, b_w)).to(mask_preds.device)
        for i in range(num_imgs):
            _, i_h, i_w = gt_masks[i].shape
            gt_masks[i] = nn.ConstantPad2d((0, b_w * self.strides[0] - i_w, 0, b_h * self.strides[0] - i_h), 0)(
                torch.tensor(gt_masks[i])).to(mask_preds.device)
            gt_masks[i] = F.upsample_bilinear(gt_masks[i].float().unsqueeze(0), (b_h, b_w))[0]
            ind_this_img = torch.nonzero(pos_img_inds == i).flatten()
            ins_this_img = pos_ins_inds[ind_this_img]
            mask_target[ind_this_img] = gt_masks[i][ins_this_img]
        loss_mask = self.dice_loss(mask_preds, mask_target) * self.dice_weight

        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask)
        '''

    def get_bboxes_cpu(self, cls_scores, mask_preds, img_metas, cfg):
        flatten_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).sigmoid()
        _, _, b_h, b_w = mask_preds[0].shape
        for i in range(5):
            mask_preds[i] = F.upsample_bilinear(mask_preds[i], (b_h, b_w))
        mask_preds = torch.cat(mask_preds, dim=1)[0]
        scores, labels = torch.max(flatten_cls_scores, dim=-1)
        nms_pre = cfg.get('nms_pre', -1)
        mask_thr = cfg.get('mask_thr_binary', -1)
        max_per_img = cfg.get('max_per_img', -1)
        crop_h, crop_w, _ = img_metas[0]['img_shape']
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        score_thr = cfg.get('score_thr', 0)
        conf_inds = scores > score_thr
        mask_preds, scores, labels = mask_preds[conf_inds], scores[conf_inds], labels[conf_inds]

        if nms_pre > 0 and scores.shape[0] > nms_pre:
            _, topk_inds = scores.topk(nms_pre)
            scores = scores[topk_inds]
            labels = labels[topk_inds]
            mask_preds = mask_preds[topk_inds]

        mask_fake = F.sigmoid(mask_preds.unsqueeze(0))[0] > mask_thr
        keeps = self.nms(scores, labels, mask_fake, cfg.nms.iou_thr)
        keeps = torch.tensor(keeps)
        scores = scores[keeps]
        labels = labels[keeps]
        mask_preds = mask_preds[keeps]

        # reduce out answer to save mem
        if max_per_img < scores.shape[0]:
            _, topk_inds = scores.topk(max_per_img)
            scores = scores[topk_inds]
            labels = labels[topk_inds]
            mask_preds = mask_preds[topk_inds]

        pp = dict()
        pp['pad_shape'] = (b_h * self.strides[0], b_w * self.strides[0])
        pp['img_shape'] = (crop_h, crop_w)
        pp['ori_shape'] = (ori_h, ori_w)
        pp['mask_thr'] = mask_thr
        img_metas[0]['cpu_postprocess'] = pp
        masks = mask_preds.sigmoid()

        n = len(masks)
        det_bboxes = np.zeros((n, 5))
        det_labels = np.zeros(n).astype(int)
        det_masks = []
        for i in range(n):
            det_bboxes[i, -1] = scores[i]
            det_labels[i] = labels[i]
            det_masks.append(masks[i])
        # det_masks = np.array(det_masks)
        det_masks = torch.stack(det_masks, dim=0)
        return det_bboxes, det_labels, det_masks

    @force_fp32(apply_to=('cls_scores', 'mask_preds_x', 'mask_preds_y'))
    def get_bboxes(self, cls_scores, mask_preds_x, mask_preds_y, img_metas, cfg, rescale=None):

        x_to_new = []
        y_to_new = []
        base = 0
        for grid in self.grid_num:
            single_x_new = torch.tensor([i for i in range(grid)]* grid).long() + base
            single_y_new = torch.tensor([i for i in range(grid) for j in range(grid) ]).long() + base
            x_to_new.append(single_x_new)
            y_to_new.append(single_y_new)
            
            base += grid
        x_to_new = torch.cat(x_to_new).flatten()
        y_to_new = torch.cat(y_to_new).flatten()

        if cfg.get('cpu_test', False):
            return self.get_bboxes_cpu(cls_scores, mask_preds, img_metas, cfg)

        flatten_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).sigmoid()
        _, _, b_h, b_w = mask_preds_x[0].shape
        for i in range(5):
            mask_preds_x[i] = F.upsample_bilinear(mask_preds_x[i], (b_h, b_w))
            mask_preds_y[i] = F.upsample_bilinear(mask_preds_y[i], (b_h, b_w))
        mask_preds_x = torch.cat(mask_preds_x, dim=1)[0]
        mask_preds_y = torch.cat(mask_preds_y, dim=1)[0]
        scores, labels = torch.max(flatten_cls_scores, dim=-1)
        nms_pre = cfg.get('nms_pre', -1)
        mask_thr = cfg.get('mask_thr_binary', -1)
        max_per_img = cfg.get('max_per_img', -1)
        crop_h, crop_w, _ = img_metas[0]['img_shape']
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        if nms_pre > 0 and scores.shape[0] > nms_pre:
            _, topk_inds = scores.topk(nms_pre)
            scores = scores[topk_inds]
            labels = labels[topk_inds]

            # 这里需要注意
            mask_preds_x = mask_preds_x[x_to_new[topk_inds]]
            mask_preds_y = mask_preds_y[y_to_new[topk_inds]]

            mask_preds = (mask_preds_x.sigmoid()).mul(mask_preds_y.sigmoid())
            #mask_preds = mask_preds[topk_inds]
            mask_fake = (mask_preds.unsqueeze(0))[0] > mask_thr
            keeps = self.nms(scores, labels, mask_fake, cfg.nms.iou_thr)
            keeps = torch.tensor(keeps)
            scores = scores[keeps]
            labels = labels[keeps]
            mask_preds = mask_preds.squeeze(0)[keeps]
            #reduce out answer to save mem
            if max_per_img < scores.shape[0]:
                _, topk_inds = scores.topk(max_per_img)
                scores = scores[topk_inds]
                labels = labels[topk_inds]
                mask_preds = mask_preds[topk_inds]

            pp = dict()
            pp['pad_shape'] = (b_h * self.strides[0], b_w * self.strides[0])
            pp['img_shape'] = (crop_h, crop_w)
            pp['ori_shape'] = (ori_h, ori_w)
            pp['mask_thr'] = mask_thr
            img_metas[0]['cpu_postprocess'] = pp
            masks = mask_preds
            
            '''
            mask_preds = F.upsample_bilinear(mask_preds.unsqueeze(0), (b_h * self.strides[0], b_w * self.strides[0]))
            mask_preds = mask_preds[:, :, :crop_h, :crop_w]
            # 先做nms再 resize
            mask_preds = mask_preds.squeeze(0)
            mask_preds = F.upsample_bilinear(mask_preds.unsqueeze(0), (ori_h, ori_w))
            #mask_preds = F.sigmoid(mask_preds)[0]
            mask_preds = mask_preds > mask_thr
            masks = mask_preds.squeeze(0)
            '''
            n = len(masks)
            det_bboxes = np.zeros((n, 5))
            det_labels = np.zeros(n).astype(int)
            det_masks = []
            for i in range(n):
                det_bboxes[i, -1] = scores[i]
                det_labels[i] = labels[i]
                det_masks.append(masks[i])
            det_masks = np.array(det_masks)
        else:
            raise RuntimeError("wroong nms_pre")
        return det_bboxes, det_labels, det_masks

    def iou_calc(self, mask1, mask2):
        overlap = mask1 & mask2
        union = mask1 | mask2
        iou = float(overlap.sum() + 1) / float(union.sum() + 1)
        return iou

    def nms(self, scores, labels, masks, iou_threshold=0.5):
        """
        nms function
        :param boxes: list of box
        :param iou_threshold:
        :return:
        """
        return_mask = []
        keeps = []
        n = len(labels)
        if n > 0:
            masks_dict = {}
            for i in range(n):
                if labels[i].item() in masks_dict:
                    masks_dict[labels[i].item()].append([masks[i], labels[i], scores[i], i])
                else:
                    masks_dict[labels[i].item()] = [[masks[i], labels[i], scores[i], i]]
            for masks in masks_dict.values():
                if len(masks) == 1:
                    return_mask.append(masks[0])
                    keeps.append(masks[0][3])
                else:
                    while (len(masks)):
                        best_mask = masks.pop(0)
                        return_mask.append(best_mask)
                        keeps.append(best_mask[3])
                        j = 0
                        for i in range(len(masks)):
                            i -= j
                            if self.iou_calc(best_mask[0], masks[i][0]) > iou_threshold:
                                masks.pop(i)
                                j += 1
        return keeps
