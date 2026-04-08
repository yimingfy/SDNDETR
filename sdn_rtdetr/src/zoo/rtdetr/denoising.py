"""by lyuwenyu
"""
import re

import numpy as np
import torch
import os
import glob
import threading

from PIL.ImageOps import scale

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou, matched_box_iou

class HardSampleGenerator:
    def  __init__(self, low_bound, high_bound, sampling_num):
        # iou bound
        self.low_bound = low_bound
        self.high_bound = high_bound
        # sampling num for each iou interval
        self.sampling_num = sampling_num

    def gene_random_noise_scale(self, xy_scale, wh_scale, rand_num):
        sxy = 2 * xy_scale * torch.rand(rand_num, 2) - xy_scale
        if wh_scale > 1:
            swh = (wh_scale + 1) * torch.rand(rand_num, 2) - 1      # we set low scale -1 for wh
        else:
            swh = 2 * wh_scale * torch.rand(rand_num, 2) - wh_scale
        rand_scales = torch.cat((sxy, swh), 1)
        return rand_scales

    def get_samples_by_iou(self, scales_collection):
        iou_width = 0.1
        interval_num = round(1.0 / iou_width)
        # add scale on norm box (0.5, 0.5, 0.1, 0.1)
        norm_box = torch.tensor([0.5, 0.5, 0.1, 0.1])
        norm_box = norm_box.repeat(scales_collection.shape[0], 1)
        diff = torch.cat((norm_box[:, 2:] * 0.5, norm_box[:, 2:]), dim=1)
        hard_box = norm_box + diff * scales_collection
        norm_box = box_cxcywh_to_xyxy(norm_box)
        hard_box = box_cxcywh_to_xyxy(hard_box)
        scales_iou = matched_box_iou(norm_box, hard_box)
        # get samples for different iou intervals
        samples = []
        enough = True
        low_idx = int(self.low_bound * interval_num)
        high_idx = int(self.high_bound * interval_num)
        for i in range(low_idx, high_idx):
            low_iou_thres = iou_width * i
            high_iou_thres = iou_width * (i + 1)
            tmp_samples_idx = (scales_iou > low_iou_thres) & (scales_iou <= high_iou_thres)
            # print(i, ": ", tmp_samples_idx.sum().item())
            if tmp_samples_idx.sum() < self.sampling_num: # if samples are not enough
                enough = False
                break
            tmp_scales = scales_collection[tmp_samples_idx]
            rand_idx = torch.randperm(tmp_scales.shape[0])[:self.sampling_num]
            tmp_scales = tmp_scales[rand_idx]
            samples.append(tmp_scales)

        samples = torch.cat(samples, 0)
        return samples, enough

    def gene_hard_samples(self):
        rand_scales_collection = torch.tensor([])
        enough_flag = False
        hard_box_samples = None
        while not enough_flag:
            rand_scales1 = self.gene_random_noise_scale(1, 1, 10 * self.sampling_num)
            rand_scales2 = self.gene_random_noise_scale(2, 2, 10 * self.sampling_num)
            rand_scales3 = self.gene_random_noise_scale(0.5, 0.5, 10 * self.sampling_num)
            # rand_scales_collection = torch.cat((rand_scales_collection, rand_scales1, rand_scales2), 0)
            # rand_scales_collection = torch.cat((rand_scales_collection, rand_scales1, rand_scales3), 0)
            rand_scales_collection = torch.cat((rand_scales_collection, rand_scales1, rand_scales2, rand_scales3), 0)
            hard_box_samples, enough_flag = self.get_samples_by_iou(rand_scales_collection)
        return hard_box_samples.to('cpu')


class MixSamplingDenoising:
    def __init__(self, num_update=5000, warm_iter=0, positive_box_noise=0.3, negative_box_noise=1.0,
                 low_bound=0.2, high_bound=0.5, hard_label_noise=0.25, fixed_label_noise=False):
        # NoiseType
        # self.noise_type = ["hsdn"]
        # self.noise_weight = [1]
        # self.noise_type = ["sdn", "dn"]
        # self.noise_weight = [1, 1]
        # self.noise_type = ["dn"]
        # self.noise_weight = [1]
        # self.noise_type = ["sdn"]
        # self.noise_weight = [1]
        self.noise_type = ["sdn", "hsdn"]
        self.noise_weight = [1, 1]
        # self.noise_type = ["sdn", "hsdn", "dn"]
        # self.noise_weight = [1, 1, 1]
        print("Denoising Methods: ", self.noise_type)
        print("Denoising Methods weights: ", self.noise_weight)
        # sdn param
        self.num_update = num_update
        self.positive_noise = positive_box_noise
        self.negative_noise = negative_box_noise
        # dynamic params
        self.num_noise = num_update * 6
        self.warm_iter = warm_iter
        self.sample_count = 0
        self.sampling_box_samples = []
        self.noise_collection = []
        self.label_noise = 0.4

        # Sampling from hard samples param
        self.hsg = None
        self.per_interval_num = 8000
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.hard_label_noise = hard_label_noise
        self.hard_box_samples = []
        self.hard_num_noise = 0
        # TODO i change this
        self.use_same_ln = False
        if self.use_same_ln:
            self.hard_label_noise = self.label_noise
        # init noise
        self.init_sampling()
        # accumulate param
        self.group_num = 0
        self.iter = 0
        # for ablation study
        self.fixed_label_noise = fixed_label_noise
        if self.fixed_label_noise:
            self.label_noise = 0.25
            self.hard_label_noise = 0.25


    def init_sampling(self):
        self.sampling_box_samples = self.gene_random_xywh_noise_scale()
        if 'hsdn' in self.noise_type:
            self.hsg = HardSampleGenerator(self.low_bound, self.high_bound, self.per_interval_num)
            self.hard_box_samples = self.hsg.gene_hard_samples()
            self.hard_num_noise = self.hard_box_samples.shape[0]

    def gene_random_xywh_noise_scale(self):
        rand_xy = 4.0 * torch.rand(self.num_noise, 2) - 2.0
        rand_wh = 3.0 * torch.rand(self.num_noise, 2) - 1.0  # wh scale >= -1
        return torch.cat((rand_xy, rand_wh), dim=1).to('cpu')

    def update_sampling_noise(self):
        # deal new noise
        print('################Updating noise#################')
        self.noise_collection = torch.cat(self.noise_collection.copy(), dim=0)
        self.sampling_box_samples, self.label_noise = self.get_sampling_noise()
        if self.fixed_label_noise:
            self.label_noise=0.25
        if self.use_same_ln:
            self.hard_label_noise = self.label_noise
        self.num_noise = self.sampling_box_samples.shape[0]
        self.sample_count = 0
        self.noise_collection = []
        print("noise num: ", self.num_noise)
        print("label noise: ", self.label_noise)
        print("avg group num: ", self.group_num / self.iter)


    def save_model_noise(self, noises, batch_size):
        self.sample_count += batch_size
        self.noise_collection.append(noises.to('cpu'))
        # update noise if counts >= num_update
        tmp_num_update = self.num_update
        if self.warm_iter > 0:
            tmp_num_update = int(self.num_update / 5)
        if self.sample_count >= tmp_num_update:
            # sdn
            self.update_sampling_noise()
            self.warm_iter = self.warm_iter - 1
            # hsdn
            if 'hsdn' in self.noise_type:
                self.hard_box_samples = self.hsg.gene_hard_samples()
                self.hard_num_noise = self.hard_box_samples.shape[0]
                print("update hard noise samples")
            print('################ End updating noise#################')

    def calculate_scale(self, box1, box2):
        cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
        cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
        scalex = 2 * (cx2 - cx1) / w1
        scaley = 2 * (cy2 - cy1) / h1
        scalew = (w2 - w1) / w1
        scaleh = (h2 - h1) / h1

        return torch.cat([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)],
                         dim=1)

    def calculate_accuracy(self, y_true, y_pred):
        """
        calculate accura
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred should keep same length")

        correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def get_sampling_noise(self):
        # calculate box and label noise
        tbox = self.noise_collection[:, 1:5]
        pbox = self.noise_collection[:, 6:]
        scales = self.calculate_scale(tbox, pbox)
        bbox_noise = scales
        cls_noise = 1 - self.calculate_accuracy(self.noise_collection[:, 0], self.noise_collection[:, 5])
        return bbox_noise, cls_noise

    def init_query_and_mask(self, bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
        input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
        input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
        pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
        for i in range(bs):
            num_gt = num_gts[i]
            if num_gt > 0:
                input_query_class[i, :num_gt] = targets[i]['labels']
                input_query_bbox[i, :num_gt] = targets[i]['boxes']
                pad_gt_mask[i, :num_gt] = 1
        # each group has positive and negative queries.
        input_query_class = input_query_class.tile([1, num_group])
        input_query_bbox = input_query_bbox.tile([1, num_group, 1])
        pad_gt_mask = pad_gt_mask.tile([1, num_group])
        return input_query_class, input_query_bbox, pad_gt_mask

    # def get_num_group(self, num_denoising, max_gt_num):
    #     num_group = num_denoising // max_gt_num
    #     num_group = 1 if num_group == 0 else num_group
    #     return num_group

    def adjust_list(self, values):
        rounded_values = [round(value) for value in values]
        for i in range(len(rounded_values)):
            if rounded_values[i] < 1:
                rounded_values[i] = 1
        return rounded_values

    def get_num_group(self, num_denoising, max_gt_num):
        num_group = num_denoising // max_gt_num
        num_group = len(self.noise_weight) if num_group == 0 else num_group
        # noise group for different type
        noise_weight = self.noise_weight.copy()
        sum_weight = sum(noise_weight)
        msdn_group = [weight * num_group/ sum_weight for weight in noise_weight]
        msdn_group = self.adjust_list(msdn_group)
        num_group = sum(msdn_group)
        return num_group, msdn_group

    def get_contrastive_mask_and_idx(self, bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
        # positive and negative mask
        negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
        negative_gt_mask[:, max_gt_num:] = 1
        negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
        positive_gt_mask = 1 - negative_gt_mask
        # contrastive denoising training positive index
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        return positive_gt_mask, negative_gt_mask, dn_positive_idx

    def get_positive_idx(self, num_gts, num_group, pad_gt_mask):
        # contrastive denoising training positive index
        positive_gt_mask = pad_gt_mask
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        return dn_positive_idx

    def get_attention_mask(self, tgt_size, num_denoising, num_group, max_gt_num, device):
        # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True
        # TODO: need to try what is good
        # attn_mask[:num_denoising, num_denoising:] = True

        # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_gt_num  * i: max_gt_num * (i + 1), max_gt_num  * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_gt_num  * i: max_gt_num  * (i + 1), :max_gt_num * i ] = True
            else:
                attn_mask[max_gt_num * i: max_gt_num * (i + 1), max_gt_num * (i + 1): num_denoising] = True
                attn_mask[max_gt_num * i: max_gt_num * (i + 1), :max_gt_num * i] = True
        return attn_mask

    # one 2 one box iou
    def batch_box_iou(self, gt_box, noised_box):
        res_iou = []
        for gt, noised in zip(gt_box, noised_box):
            res_iou.append(matched_box_iou(gt, noised).unsqueeze(0))
        return torch.cat(res_iou, dim=0)


    def add_label_noise(self, input_query_class, num_classes, pad_gt_mask):
        noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.label_noise

        # randomly put a new one here
        label_offset = torch.randint_like(noise_mask, 1, num_classes, dtype=input_query_class.dtype)
        new_label = (input_query_class + label_offset) % num_classes
        input_query_class = torch.where(noise_mask & pad_gt_mask, new_label, input_query_class)

        return input_query_class

    def add_relate_box_noise(self, input_query_bbox, num_denoising):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_samples[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # get positive and negative noise
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        positive_rand_part *= rand_sign
        # negative_rand_part *= rand_sign
        positive_noise_scale = (1 + positive_rand_part) * noise_scale
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        # add noise to known bbox
        input_query_bbox = positive_query_bbox
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_sdn_query(self, input_query_class, num_classes, input_query_bbox,
                         pad_gt_mask):
        num_denoising = input_query_bbox.shape[1]
        input_query_class = self.add_label_noise(input_query_class, num_classes, pad_gt_mask)
        input_query_bbox = self.add_relate_box_noise(input_query_bbox, num_denoising)
        return input_query_class, input_query_bbox

    # hsdn noise
    def add_hsdn_label_noise(self, input_query_class, num_classes, pad_gt_mask):
        noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.hard_label_noise
        # randomly put a new one here
        label_offset = torch.randint_like(noise_mask, 1, num_classes, dtype=input_query_class.dtype)
        new_label = (input_query_class + label_offset) % num_classes
        input_query_class = torch.where(noise_mask & pad_gt_mask, new_label, input_query_class)

        return input_query_class

    def add_hsdn_box_noise(self, input_query_bbox, num_denoising):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.hard_num_noise, size=(num_denoising,)) for _ in range(bs)]
        noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.hard_box_samples[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # get positive and negative noise
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        positive_rand_part *= rand_sign
        positive_noise_scale = (1 + positive_rand_part) * noise_scale
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff

        # add noise to known bbox
        input_query_bbox = positive_query_bbox
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        # noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_hsdn_query(self, input_query_class, num_classes, input_query_bbox, pad_gt_mask):
        num_denoising = input_query_bbox.shape[1]
        input_query_class = self.add_hsdn_label_noise(input_query_class, num_classes, pad_gt_mask)
        input_query_bbox = self.add_hsdn_box_noise(input_query_bbox, num_denoising)
        return input_query_class, input_query_bbox

    # dn noise
    def add_dn_label_noise(self, input_query_class, noise_ratio, num_classes, pad_gt_mask):
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
        return input_query_class

    def add_dn_box_noise(self, input_query_bbox, noise_scale):
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_dn_query(self, input_query_class, num_classes, input_query_bbox, cls_noise_ratio, box_noise_scale,
                    pad_gt_mask):
        input_query_class = self.add_dn_label_noise(input_query_class, cls_noise_ratio, num_classes, pad_gt_mask)
        input_query_bbox = self.add_dn_box_noise(input_query_bbox, box_noise_scale)
        return input_query_class, input_query_bbox

    def get_sampling_cdn_group(self, targets, num_classes, num_queries, class_embed, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
        """
        Get Mixed Sampling Contrastive Denoising
        This denoising is true denoising, not need x 2
        """
        # step0: init params
        if num_denoising <= 0:
            return None, None, None, None

        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        max_gt_num = max(num_gts)
        if max_gt_num == 0:
            return None, None, None, None
        num_group, msdn_group = self.get_num_group(num_denoising, max_gt_num)
        self.group_num += num_group
        self.iter += 1
        # pad gt to max_num of a batch
        bs = len(num_gts)

        # step1: get contrastive group and mask
        input_query_class, input_query_bbox, pad_gt_mask = self.init_query_and_mask(bs, max_gt_num, num_classes,
                                                                               num_gts, targets, num_group, device)
        # step2: get mask and positive idx
        dn_positive_idx = self.get_positive_idx(num_gts, num_group, pad_gt_mask)

        # step3: add noise to box and class
        num_denoising = int(max_gt_num * num_group)  # total denoising queries
        # 这里写一个循环，一次一次的放入噪声组
        start_idx = 0
        for i in range(len(self.noise_type)):
            if i != 0:
                start_idx += int(msdn_group[i - 1] * max_gt_num)
            end_idx = start_idx + int(msdn_group[i] * max_gt_num)
            if self.noise_type[i] == "sdn" and msdn_group[i] > 0:

                tmp_input_query_class, tmp_input_query_bbox = self.get_sdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 pad_gt_mask[:, start_idx:end_idx])
            elif self.noise_type[i] == "hsdn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_hsdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 pad_gt_mask[:, start_idx:end_idx])

            elif self.noise_type[i] == "dn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_dn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 label_noise_ratio,
                                                                                 box_noise_scale,
                                                                                 pad_gt_mask[:, start_idx:end_idx],
                                                                                 )


            else:
                tmp_input_query_class, tmp_input_query_bbox = None, None

            if msdn_group[i] > 0:
                input_query_class[:, start_idx:end_idx] = tmp_input_query_class
                input_query_bbox[:, start_idx:end_idx] = tmp_input_query_bbox

        input_query_class = class_embed(input_query_class)


        # step4: get attention mask
        tgt_size = num_denoising + num_queries
        attn_mask = self.get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

        dn_meta = {
            "dn_positive_idx": dn_positive_idx,
            "dn_num_group": num_group,
            "dn_num_split": [num_denoising, num_queries]
        }

        return input_query_class, input_query_bbox, attn_mask, dn_meta



# 混合去噪，把label也放到一起
class MixDenoising:
    def __init__(self, num_update=5000, num_noise=25000, positive_box_noise=0.25, negative_box_noise=0.5,
                 negative_label_noise=0.1):
        # NoiseType
        # self.noise_type = ["ssdn", "tsdn", "cdn"]
        # self.noise_weight = [1, 1, 1]
        # self.noise_type = ["sdn", "cdn"]
        # self.noise_weight = [1, 1]

        self.noise_type = ["sdn"]
        self.noise_weight = [1]
        print("Denoising Methods: ", self.noise_type)
        print("Denoising Methods weights: ", self.noise_weight)
        # sdn param
        self.num_update = num_update
        self.num_noise = num_noise
        self.positive_noise = positive_box_noise
        self.negative_noise = negative_box_noise
        self.negative_label_noise = negative_label_noise
        # dynamic params
        self.sample_count = 0
        self.sampling_label_noise = []
        self.sampling_box_noise = []
        self.noise_collection = []

        # init noise
        self.init_sampling()

    def gene_random_xywh_noise_scale(self):
        rand_box_scales = 3.0 * torch.rand(self.num_noise, 4) - 1.0
        rand_label_sign = torch.rand(self.num_noise) > 0.9   # add large label noise
        return rand_box_scales, rand_label_sign

    def init_sampling(self):
        self.sampling_box_noise, self.sampling_label_noise = self.gene_random_xywh_noise_scale()

    def update_sampling_noise(self):
        # deal new noise
        print('################Updating noise#################')
        self.noise_collection = torch.cat(self.noise_collection.copy(), dim=0)
        self.sampling_box_noise, self.sampling_label_noise = self.get_sampling_noise()
        self.num_noise = self.sampling_box_noise.shape[0]
        # clear collection
        self.sample_count = 0
        self.noise_collection = []
        print("noise num: ", self.num_noise)
        print('################ End updating noise#################')

    def save_model_noise(self, noises, batch_size):
        self.sample_count += batch_size
        self.noise_collection.append(noises.to('cpu'))
        # update noise if counts >= num_update
        if self.sample_count >= self.num_update:
            self.update_sampling_noise()

    def calculate_scale(self, box1, box2):
        cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
        cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
        scalex = 2 * (cx2 - cx1) / w1
        scaley = 2 * (cy2 - cy1) / h1
        scalew = (w2 - w1) / w1
        scaleh = (h2 - h1) / h1
        return torch.cat([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)],
                         dim=1)

    def calculate_label_sign(self, y_true, y_pred):
        label_sign = y_true.int() == y_pred.int()   # True means box label right
        return label_sign

    def calculate_accuracy(self, y_true, y_pred):
        """
        calculate accura
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred should keep same length")

        correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def get_sampling_noise(self):
        # filtering boxes
        # self.sampling_box_noise = torch.tensor(self.sampling_box_noise)
        tbox = self.noise_collection[:, 1:5]
        pbox = self.noise_collection[:, 6:]
        scales = self.calculate_scale(tbox, pbox)
        bbox_noise = scales
        label_noise = self.calculate_label_sign(self.noise_collection[:, 0], self.noise_collection[:, 5])
        return bbox_noise, label_noise

    def init_query_and_mask(self, bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
        input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
        input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
        pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
        for i in range(bs):
            num_gt = num_gts[i]
            if num_gt > 0:
                input_query_class[i, :num_gt] = targets[i]['labels']
                input_query_bbox[i, :num_gt] = targets[i]['boxes']
                pad_gt_mask[i, :num_gt] = 1
        # each group has positive and negative queries.
        input_query_class = input_query_class.tile([1, 2 * num_group])
        input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
        pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
        return input_query_class, input_query_bbox, pad_gt_mask

    def adjust_list(self, values):
        rounded_values = [round(value) for value in values]
        # every denoising have more than one group
        for i in range(len(rounded_values)):
            if rounded_values[i] < 1:
                rounded_values[i] = 1
        return rounded_values

    def get_num_group(self, num_denoising, max_gt_num):
        num_group = num_denoising // max_gt_num
        num_group = len(self.noise_weight) if num_group == 0 else num_group
        # noise group for different type
        noise_weight = self.noise_weight.copy()
        sum_weight = sum(noise_weight)
        msdn_group = [weight * num_group/ sum_weight for weight in noise_weight]
        msdn_group = self.adjust_list(msdn_group)
        num_group = sum(msdn_group)
        return num_group, msdn_group

    def get_contrastive_mask_and_idx(self, bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
        # positive and negative mask
        negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
        negative_gt_mask[:, max_gt_num:] = 1
        negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
        positive_gt_mask = 1 - negative_gt_mask
        # contrastive denoising training positive index
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        return positive_gt_mask, negative_gt_mask, dn_positive_idx

    def get_attention_mask(self, tgt_size, num_denoising, num_group, max_gt_num, device):
        # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True

        # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
            else:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        return attn_mask

    def solve_error_labels(self, noise_mask, max_gt_num, pad_gt_mask):
        # only save useful mask
        noise_mask = noise_mask & pad_gt_mask
        split_noise_mask = torch.split(noise_mask, max_gt_num, dim=1)
        positive_mask = split_noise_mask[0::2]
        negative_mask = split_noise_mask[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_mask, ng_mask) in enumerate(zip(positive_mask, negative_mask)):
            label_errors = pos_mask > ng_mask  # error when pos_iou > ng_iou(True > False)
            # label_errors = label_errors.to(torch.bool)
            image_id, idx = torch.where(label_errors)     # it is a torch.bool tensor
            if idx.numel() != 0:
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组，直接让他等于false，不改变正样本噪声比例
        for idx in error_idx:
            noise_mask[idx[0], idx[1] + max_gt_num] = noise_mask[idx[0], idx[1]].clone()
        return noise_mask

    # one 2 one box iou
    def batch_box_iou(self, gt_box, noised_box):
        res_iou = []
        for gt, noised in zip(gt_box, noised_box):
            res_iou.append(matched_box_iou(gt, noised).unsqueeze(0))
        return torch.cat(res_iou, dim=0)

    def solve_error_bbox(self, gt_box, noised_box, max_gt_num, pad_gt_mask):
        noised_iou = self.batch_box_iou(gt_box, noised_box)
        # 去掉无关的
        noised_iou = noised_iou.masked_fill(~pad_gt_mask, 0.0)
        split_iou = torch.split(noised_iou, max_gt_num, dim=1)
        positive_iou = split_iou[0::2]
        negative_iou = split_iou[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_iou, ng_iou) in enumerate(zip(positive_iou, negative_iou)):
            box_errors = pos_iou < ng_iou  # error when pos_iou < ng_iou
            # box_errors = torch.tensor(box_errors, dtype=torch.bool)
            image_id, idx = torch.where(box_errors)
            if idx.numel() != 0:
                # print('error box appear')
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组,swap一下
        for idx in error_idx:
            pos_box = noised_box[idx[0], idx[1]].clone()
            ng_box = noised_box[idx[0], idx[1] + max_gt_num].clone()
            noised_box[idx[0], idx[1]], noised_box[idx[0], idx[1] + max_gt_num] = ng_box, pos_box
        return noised_box

    # sdn noise
    def add_label_noise(self, input_query_class, noise_idx, max_gt_num, num_classes, pad_gt_mask, negative_gt_mask):
        # need to debug
        noise_label = []
        for idx in noise_idx:
            tmp_noise = self.sampling_label_noise[idx].clone()
            noise_label.append(tmp_noise.unsqueeze(0))
        noise_label = torch.cat(noise_label, dim=0).to(input_query_class.device)
        positive_noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.label_noise
        negative_noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < (2 * self.label_noise)
        positive_label_noise = ~noise_label | positive_noise_mask
        negative_label_noise = ~torch.roll(noise_label, shifts=max_gt_num, dims=1) | negative_noise_mask
        noise_mask = torch.where(negative_gt_mask.squeeze(-1) == 1.0, negative_label_noise, positive_label_noise)
        # solve error labels
        noise_mask = self.solve_error_labels(noise_mask, max_gt_num, pad_gt_mask)
        # randomly put a new one here(use offset)
        label_offset = torch.randint_like(noise_mask, 1, num_classes, dtype=input_query_class.dtype)
        new_label = (input_query_class + label_offset) % num_classes
        input_query_class = torch.where(noise_mask, new_label, input_query_class)
        return input_query_class

    # sdn only negative noise
    def add_negative_label_noise(self, input_query_class, noise_idx, max_gt_num, num_classes, pad_gt_mask, negative_gt_mask):
        # need to debug
        noise_label = []
        for idx in noise_idx:
            tmp_noise = self.sampling_label_noise[idx].clone()
            noise_label.append(tmp_noise.unsqueeze(0))
        noise_label = torch.cat(noise_label, dim=0).to(input_query_class.device)
        negative_noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.negative_label_noise
        negative_noise_mask = ~torch.roll(noise_label, shifts=max_gt_num, dims=1) | negative_noise_mask
        noise_mask = torch.where(negative_gt_mask.squeeze(-1) == 1.0, negative_noise_mask, noise_label)
        # solve error labels
        # noise_mask = self.solve_error_labels(noise_mask, max_gt_num, pad_gt_mask)
        # randomly put a new one here(use offset)
        label_offset = torch.randint_like(noise_mask, 1, num_classes, dtype=input_query_class.dtype)
        new_label = (input_query_class + label_offset) % num_classes
        input_query_class = torch.where(noise_mask, new_label, input_query_class)
        return input_query_class

    def add_box_noise(self, input_query_bbox, noise_idx, negative_gt_mask, max_gt_num, pad_gt_mask):
        noise_scale = []
        for idx in noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # get positive and negative noise
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = torch.rand_like(input_query_bbox) * self.negative_noise + self.positive_noise
        positive_rand_part *= rand_sign
        # negative_rand_part *= rand_sign
        positive_noise_scale = (1 + positive_rand_part) * noise_scale
        negative_noise_scale = torch.roll(noise_scale, shifts=max_gt_num,
                                          dims=1) * (1 + negative_rand_part)  # add to same noise
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_noise_scale * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_sdn_query(self, input_query_class, num_classes, input_query_bbox,
                       negative_gt_mask, max_gt_num, pad_gt_mask):
        num_denoising = input_query_bbox.shape[1]
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        input_query_class = self.add_negative_label_noise(input_query_class, noise_idx, max_gt_num, num_classes,
                                                 pad_gt_mask, negative_gt_mask)
        input_query_bbox = self.add_box_noise(input_query_bbox, noise_idx, negative_gt_mask,
                                                      max_gt_num, pad_gt_mask)
        return input_query_class, input_query_bbox

    # cdn noise
    def add_cdn_label_noise(self, input_query_class, noise_ratio, num_classes, pad_gt_mask):
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
        return input_query_class

    def add_cdn_box_noise(self, input_query_bbox, noise_scale, negative_gt_mask):
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_cdn_query(self, input_query_class, num_classes, input_query_bbox, cls_noise_ratio, box_noise_scale,
                      negative_gt_mask, pad_gt_mask):
        input_query_class = self.add_cdn_label_noise(input_query_class, cls_noise_ratio, num_classes, pad_gt_mask)
        input_query_bbox = self.add_cdn_box_noise(input_query_bbox, box_noise_scale, negative_gt_mask)
        return input_query_class, input_query_bbox

    # # shuffle noise in each group, will be restore in calculate loss 这个不方便，不如直接改变positive idx
    # def shuffle_noise_v1(self, input_query_class, input_query_bbox, max_gt_num, num_group):
    #     shuffle_indices = []
    #     group_size = max_gt_num * 2
    #     # shuffle groups
    #     for i in range(num_group):
    #         start_idx = i * group_size
    #         end_idx = start_idx + group_size
    #         group_class = input_query_class[:, start_idx:end_idx]
    #         group_bbox = input_query_bbox[:, start_idx:end_idx]
    #         shuffled_group_indices = torch.randperm(group_size)
    #         input_query_class[:, start_idx:end_idx] = group_class[:, shuffled_group_indices]
    #         input_query_bbox[:, start_idx:end_idx] = group_bbox[:, shuffled_group_indices]
    #         shuffle_indices.append(shuffled_group_indices)
    #     return input_query_class, input_query_bbox, shuffle_indices
    # shuffle noise in each group, will be restore in calculate loss
    def shuffle_noise(self, input_query_class, input_query_bbox, dn_positive_idx, max_gt_num, num_group):
        idx_map = []
        group_size = max_gt_num * 2
        # shuffle groups
        for i in range(num_group):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            # shuffle queries
            group_class = input_query_class[:, start_idx:end_idx]
            group_bbox = input_query_bbox[:, start_idx:end_idx]
            shuffled_group_indices = torch.randperm(group_size)
            input_query_class[:, start_idx:end_idx] = group_class[:, shuffled_group_indices]
            input_query_bbox[:, start_idx:end_idx] = group_bbox[:, shuffled_group_indices]
            # get idx map
            index_map = torch.zeros_like(shuffled_group_indices, dtype=torch.int64)
            index_map[shuffled_group_indices] = torch.arange(index_map.size(0)) + start_idx
            idx_map.append(index_map)
        # map the positive shuffle_indices
        idx_map = torch.cat(idx_map, dim=0).to(input_query_class.device)
        shuffle_positive_idx = []
        for tmp_positive_idx in dn_positive_idx:
            tmp_shuffle_idx = idx_map[tmp_positive_idx]
            shuffle_positive_idx.append(tmp_shuffle_idx)

        return input_query_class, input_query_bbox, shuffle_positive_idx

    def get_sampling_cdn_group(self, targets,  num_classes, num_queries, class_embed, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
        """
        Get Mixed Sampling Contrastive Denoising
        """
        # step0: init params
        if num_denoising <= 0:
            return None, None, None, None

        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        max_gt_num = max(num_gts)
        if max_gt_num == 0:
            return None, None, None, None
        num_group, msdn_group = self.get_num_group(num_denoising, max_gt_num)
        # pad gt to max_num of a batch
        bs = len(num_gts)

        # step1: get contrastive group and mask
        input_query_class, input_query_bbox, pad_gt_mask = self.init_query_and_mask(bs, max_gt_num, num_classes,
                                                                               num_gts, targets, num_group, device)
        # step2: get mask and positive idx
        positive_gt_mask, negative_gt_mask, dn_positive_idx = (
            self.get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device))

        # step3: add noise to box and class
        num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries
        # TODO: 根据num_group分组，比如一组是ssdn，一组是tsdn，一组是cdn这样（当然我也想尝试直接按照全部的概率划分，到时候看看哪个好吧）
        # 这里写一个循环，一次一次的放入噪声组
        start_idx = 0
        for i in range(len(self.noise_type)):
            if i != 0:
                start_idx += int(msdn_group[i - 1] * 2 * max_gt_num)
            end_idx = start_idx + int(msdn_group[i] * 2 * max_gt_num)
            if self.noise_type[i] == "sdn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_sdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 negative_gt_mask[:, start_idx:end_idx],
                                                                                 max_gt_num,
                                                                                 pad_gt_mask[:, start_idx:end_idx])

            elif self.noise_type[i] == "cdn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_cdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 label_noise_ratio,
                                                                                 box_noise_scale,
                                                                                 negative_gt_mask[:, start_idx:end_idx],
                                                                                 pad_gt_mask[:, start_idx:end_idx])

            else:
                tmp_input_query_class, tmp_input_query_bbox = None, None

            if msdn_group[i] > 0:
                input_query_class[:, start_idx:end_idx] = tmp_input_query_class
                input_query_bbox[:, start_idx:end_idx] = tmp_input_query_bbox

        # step3.2: shuffle noise
        # input_query_class, input_query_bbox, shuffle_positive_idx = self.shuffle_noise(input_query_class, input_query_bbox, dn_positive_idx,
        #                                                          max_gt_num, num_group)
        input_query_class = class_embed(input_query_class)


        # step4: get attention mask
        tgt_size = num_denoising + num_queries
        attn_mask = self.get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

        dn_meta = {
            "dn_positive_idx": dn_positive_idx,
            "dn_num_group": num_group,
            "dn_num_split": [num_denoising, num_queries]
        }

        return input_query_class, input_query_bbox, attn_mask, dn_meta


# An imply only work on single gpu
# 修改一下噪声的参数
class MixSDNSamplingCDN:
    def __init__(self, num_update=5000, num_noise=25000, positive_box_noise=0.3, negative_box_noise=1.0,
                 random_label_noise=0, bias_dir=None):
        # NoiseType
        # self.noise_type = ["ssdn", "tsdn", "cdn"]
        # self.noise_weight = [1, 1, 1]
        self.noise_type = ["sdn", "cdn"]
        self.noise_weight = [1, 1]

        # self.noise_type = ["sdn"]
        # self.noise_weight = [1]
        print("Denoising Methods: ", self.noise_type)
        print("Denoising Methods weights: ", self.noise_weight)
        # sdn param
        self.num_update = num_update
        self.num_noise = num_noise
        self.positive_noise = positive_box_noise
        self.negative_noise = negative_box_noise
        # dynamic params
        self.sample_count = 0
        self.sampling_box_noise = []
        self.noise_collection = []
        self.label_noise = 0.8
        # init noise
        self.init_sampling(bias_dir)

    def gene_random_xywh_noise_scale(self):
        rand_scales = 3.0 * torch.rand(self.num_noise, 4) - 1.0
        return rand_scales

    def init_sampling(self, bias_dir=None):
        # TODO： solve deal bias file
        if bias_dir:    # load coco_bias_dir
            assert 1, "can not deal bias_file now"
            pass
        else:   # use random noise like cdn
            self.sampling_box_noise = self.gene_random_xywh_noise_scale()

    def update_sampling_noise(self):
        # deal new noise
        print('################Updating noise#################')
        self.noise_collection = torch.cat(self.noise_collection.copy(), dim=0)
        # TODO: update box noise and label noise
        self.sampling_box_noise, self.label_noise = self.get_sampling_noise()
        self.num_noise = self.sampling_box_noise.shape[0]
        self.sample_count = 0
        self.noise_collection = []
        print("noise num: ", self.num_noise)
        print('################ End updating noise#################')

    def save_model_noise(self, noises, batch_size):
        self.sample_count += batch_size
        self.noise_collection.append(noises.to('cpu'))
        # update noise if counts >= num_update
        if self.sample_count >= self.num_update:
            # print("updating noise")
            self.update_sampling_noise()

    def calculate_scale(self, box1, box2):
        cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
        cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
        scalex = 2 * (cx2 - cx1) / w1
        scaley = 2 * (cy2 - cy1) / h1
        scalew = (w2 - w1) / w1
        scaleh = (h2 - h1) / h1

        return torch.cat([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)],
                         dim=1)

    def calculate_accuracy(self, y_true, y_pred):
        """
        calculate accura
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred should keep same length")

        correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def get_sampling_noise(self):
        # filtering boxes
        # self.sampling_box_noise = torch.tensor(self.sampling_box_noise)
        tbox = self.noise_collection[:, 1:5]
        pbox = self.noise_collection[:, 6:]
        scales = self.calculate_scale(tbox, pbox)
        bbox_noise = scales
        cls_noise = 1 - self.calculate_accuracy(self.noise_collection[:, 0], self.noise_collection[:, 5])
        return bbox_noise, cls_noise

    def init_query_and_mask(self, bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
        input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
        input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
        pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
        for i in range(bs):
            num_gt = num_gts[i]
            if num_gt > 0:
                input_query_class[i, :num_gt] = targets[i]['labels']
                input_query_bbox[i, :num_gt] = targets[i]['boxes']
                pad_gt_mask[i, :num_gt] = 1
        # each group has positive and negative queries.
        input_query_class = input_query_class.tile([1, 2 * num_group])
        input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
        pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
        return input_query_class, input_query_bbox, pad_gt_mask

    def solve_error_labels(self, noise_mask, max_gt_num, pad_gt_mask):
        # only save useful mask
        noise_mask = noise_mask & pad_gt_mask
        split_noise_mask = torch.split(noise_mask, max_gt_num, dim=1)
        positive_mask = split_noise_mask[0::2]
        negative_mask = split_noise_mask[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_mask, ng_mask) in enumerate(zip(positive_mask, negative_mask)):
            label_errors = pos_mask > ng_mask  # error when pos_iou > ng_iou(True > False)
            # label_errors = label_errors.to(torch.bool)
            image_id, idx = torch.where(label_errors)     # it is a torch.bool tensor
            if idx.numel() != 0:
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组,swap一下
        for idx in error_idx:
            pos_box = noise_mask[idx[0], idx[1]].clone()
            ng_box = noise_mask[idx[0], idx[1] + max_gt_num].clone()
            noise_mask[idx[0], idx[1]], noise_mask[idx[0], idx[1] + max_gt_num] = ng_box, pos_box

        return noise_mask

    # one 2 one box iou
    def batch_box_iou(self, gt_box, noised_box):
        res_iou = []
        for gt, noised in zip(gt_box, noised_box):
            res_iou.append(matched_box_iou(gt, noised).unsqueeze(0))
        return torch.cat(res_iou, dim=0)

    def solve_error_bbox(self, gt_box, noised_box, max_gt_num, pad_gt_mask):
        noised_iou = self.batch_box_iou(gt_box, noised_box)
        # 去掉无关的
        noised_iou = noised_iou.masked_fill(~pad_gt_mask, 0.0)
        split_iou = torch.split(noised_iou, max_gt_num, dim=1)
        positive_iou = split_iou[0::2]
        negative_iou = split_iou[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_iou, ng_iou) in enumerate(zip(positive_iou, negative_iou)):
            box_errors = pos_iou < ng_iou  # error when pos_iou < ng_iou
            # box_errors = torch.tensor(box_errors, dtype=torch.bool)
            image_id, idx = torch.where(box_errors)
            if idx.numel() != 0:
                # print('error box appear')
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组,swap一下
        for idx in error_idx:
            pos_box = noised_box[idx[0], idx[1]].clone()
            ng_box = noised_box[idx[0], idx[1] + max_gt_num].clone()
            noised_box[idx[0], idx[1]], noised_box[idx[0], idx[1] + max_gt_num] = ng_box, pos_box
        return noised_box

    def add_label_noise(self, input_query_class, max_gt_num, num_classes, pad_gt_mask):
        noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.label_noise
        # todo: how to make sure negative box is worse than positive box 副样本有点太差了，需要更好的样本
        # 如果发现positive组有噪声，就和negative组交换
        noise_mask = self.solve_error_labels(noise_mask, max_gt_num, pad_gt_mask)
        # randomly put a new one here
        new_label = torch.randint_like(noise_mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(noise_mask, new_label, input_query_class)

        return input_query_class

    # TODO: use iou_based sampling to get hard samples
    def add_hard_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # positive_noise_scale = torch.cat([torch.tensor(self.sampling_box_noise[idx]).unsqueeze(0)
        #                                   for idx in batch_noise_idx], dim=0)
        # todo: how to make sure negative box is worse than positive box
        # add random positive and negative noise, this negative is not a good method, need iou
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0     # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = torch.rand_like(input_query_bbox) * self.negative_noise + self.positive_noise    # 有问题em需要让保持nage>positive
        positive_rand_part *= rand_sign
        negative_rand_part *= rand_sign
        positive_noise_scale = noise_scale + positive_rand_part
        negative_noise_scale = torch.roll(noise_scale, shifts=max_gt_num, dims=1) + negative_rand_part  # add to same noise
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_noise_scale * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def add_relate_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # get positive and negative noise
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = torch.rand_like(input_query_bbox) * self.negative_noise + self.positive_noise
        positive_rand_part *= rand_sign
        # negative_rand_part *= rand_sign
        positive_noise_scale = (1 + positive_rand_part) * noise_scale
        negative_noise_scale = torch.roll(noise_scale, shifts=max_gt_num,
                                          dims=1) * (1 + negative_rand_part)  # add to same noise
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_noise_scale * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def add_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        positive_noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            positive_noise_scale.append(tmp_noise.unsqueeze(0))
        positive_noise_scale = torch.cat(positive_noise_scale, dim=0).to(input_query_bbox.device)
        # positive_noise_scale = torch.cat([torch.tensor(self.sampling_box_noise[idx]).unsqueeze(0)
        #                                   for idx in batch_noise_idx], dim=0)
        # todo: how to make sure negative box is worse than positive box
        # add random positive and negative noise, this negative is not a good method, need iou
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0     # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = (torch.rand_like(input_query_bbox) + 1.0) * negative_gt_mask
        positive_rand_part *= rand_sign
        negative_rand_part *= rand_sign
        positive_noise_scale += positive_rand_part
        # get diff: diff is w / 2 or h / 2
        # diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2])
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_rand_part * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # input_query_bbox = torch.where(negative_gt_mask, negative_known_bbox, input_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    # def get_num_group(self, num_denoising, max_gt_num):
    #     num_group = num_denoising // max_gt_num
    #     num_group = 1 if num_group == 0 else num_group
    #     return num_group

    def adjust_list(self, values):
        # 四舍五入列表中的每个元素
        rounded_values = [round(value) for value in values]
        # 确保至少有一个噪声组
        for i in range(len(rounded_values)):
            if rounded_values[i] < 1:
                rounded_values[i] = 1
        # if rounded_values[0] < 0:
        #     rounded_values[0] = 1
        return rounded_values

    def get_num_group(self, num_denoising, max_gt_num):
        num_group = num_denoising // max_gt_num
        num_group = len(self.noise_weight) if num_group == 0 else num_group
        # noise group for different type
        noise_weight = self.noise_weight.copy()
        sum_weight = sum(noise_weight)
        msdn_group = [weight * num_group/ sum_weight for weight in noise_weight]
        msdn_group = self.adjust_list(msdn_group)
        num_group = sum(msdn_group)
        return num_group, msdn_group

    def get_contrastive_mask_and_idx(self, bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
        # positive and negative mask
        negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
        negative_gt_mask[:, max_gt_num:] = 1
        negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
        positive_gt_mask = 1 - negative_gt_mask
        # contrastive denoising training positive index
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        return positive_gt_mask, negative_gt_mask, dn_positive_idx

    def get_attention_mask(self, tgt_size, num_denoising, num_group, max_gt_num, device):
        # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True

        # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
            else:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        return attn_mask

    # ssdn noise
    def get_sdn_query(self, input_query_class, num_classes, input_query_bbox,
                       negative_gt_mask, max_gt_num, pad_gt_mask):
        num_denoising = input_query_bbox.shape[1]
        input_query_class = self.add_label_noise(input_query_class, max_gt_num, num_classes, pad_gt_mask)
        input_query_bbox = self.add_relate_box_noise(input_query_bbox, num_denoising, negative_gt_mask,
                                                      max_gt_num, pad_gt_mask)
        return input_query_class, input_query_bbox

    # cdn noise
    def add_cdn_label_noise(self, input_query_class, noise_ratio, num_classes, pad_gt_mask):
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
        return input_query_class

    def add_cdn_box_noise(self, input_query_bbox, noise_scale, negative_gt_mask):
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_cdn_query(self, input_query_class, num_classes, input_query_bbox, cls_noise_ratio, box_noise_scale,
                      negative_gt_mask, pad_gt_mask):
        input_query_class = self.add_cdn_label_noise(input_query_class, cls_noise_ratio, num_classes, pad_gt_mask)
        input_query_bbox = self.add_cdn_box_noise(input_query_bbox, box_noise_scale, negative_gt_mask)
        return input_query_class, input_query_bbox


    def get_sampling_cdn_group(self, targets,  num_classes, num_queries, class_embed, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
        """
        Get Mixed Sampling Contrastive Denoising
        """
        # step0: init params
        if num_denoising <= 0:
            return None, None, None, None

        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        max_gt_num = max(num_gts)
        if max_gt_num == 0:
            return None, None, None, None
        num_group, msdn_group = self.get_num_group(num_denoising, max_gt_num)
        # pad gt to max_num of a batch
        bs = len(num_gts)

        # step1: get contrastive group and mask
        input_query_class, input_query_bbox, pad_gt_mask = self.init_query_and_mask(bs, max_gt_num, num_classes,
                                                                               num_gts, targets, num_group, device)
        # step2: get mask and positive idx
        positive_gt_mask, negative_gt_mask, dn_positive_idx = (
            self.get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device))

        # step3: add noise to box and class
        num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries
        # TODO: 根据num_group分组，比如一组是ssdn，一组是tsdn，一组是cdn这样（当然我也想尝试直接按照全部的概率划分，到时候看看哪个好吧）
        # 这里写一个循环，一次一次的放入噪声组
        start_idx = 0
        for i in range(len(self.noise_type)):
            if i != 0:
                start_idx += int(msdn_group[i - 1] * 2 * max_gt_num)
            end_idx = start_idx + int(msdn_group[i] * 2 * max_gt_num)
            if self.noise_type[i] == "sdn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_sdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 negative_gt_mask[:, start_idx:end_idx],
                                                                                 max_gt_num,
                                                                                 pad_gt_mask[:, start_idx:end_idx])

            elif self.noise_type[i] == "cdn" and msdn_group[i] > 0:
                tmp_input_query_class, tmp_input_query_bbox = self.get_cdn_query(input_query_class[:, start_idx:end_idx].clone(),
                                                                                 num_classes,
                                                                                 input_query_bbox[:, start_idx:end_idx].clone(),
                                                                                 label_noise_ratio,
                                                                                 box_noise_scale,
                                                                                 negative_gt_mask[:, start_idx:end_idx],
                                                                                 pad_gt_mask[:, start_idx:end_idx])

            else:
                tmp_input_query_class, tmp_input_query_bbox = None, None

            if msdn_group[i] > 0:
                input_query_class[:, start_idx:end_idx] = tmp_input_query_class
                input_query_bbox[:, start_idx:end_idx] = tmp_input_query_bbox

        input_query_class = class_embed(input_query_class)
        # step4: get attention mask
        tgt_size = num_denoising + num_queries
        attn_mask = self.get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

        dn_meta = {
            "dn_positive_idx": dn_positive_idx,
            "dn_num_group": num_group,
            "dn_num_split": [num_denoising, num_queries]
        }

        return input_query_class, input_query_bbox, attn_mask, dn_meta



# TODO: there are many problems to solve, like wrong boxes in fu yang ben负样本
def get_contrastive_denoising_training_group_old(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_class, input_query_bbox, attn_mask, dn_meta


def get_num_group(num_denoising, max_gt_num):
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    return num_group


# get one group of query and mask
def init_query_and_mask(bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    return input_query_class, input_query_bbox, pad_gt_mask


def add_label_noise(input_query_class, noise_ratio, num_classes, pad_gt_mask):
    mask = torch.rand_like(input_query_class, dtype=torch.float) < (noise_ratio * 0.5)
    # randomly put a new one here
    new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
    input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
    return input_query_class


def add_box_noise(input_query_bbox, noise_scale, negative_gt_mask):
    known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
    diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * noise_scale
    rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
    rand_part = torch.rand_like(input_query_bbox)
    rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
    rand_part *= rand_sign
    known_bbox += rand_part * diff
    known_bbox.clip_(min=0.0, max=1.0)
    input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
    input_query_bbox = inverse_sigmoid(input_query_bbox)
    return input_query_bbox


def get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    return positive_gt_mask, negative_gt_mask, dn_positive_idx


# calculate the attention mask for denoising groups
def get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device):
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    return attn_mask



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,
                                             ):
    """
    Sampling Contrastive Denoising
    """
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = get_num_group(num_denoising, max_gt_num)

    # pad gt to max_num of a batch
    bs = len(num_gts)
    # get contrastive group and mask
    input_query_class, input_query_bbox, pad_gt_mask = init_query_and_mask(bs, max_gt_num, num_classes,
                                                                           num_gts, targets, num_group, device)

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    # add noise to box and class
    num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries
    if label_noise_ratio > 0:
        input_query_class = add_label_noise(input_query_class.clone(), label_noise_ratio, num_classes, pad_gt_mask)

    if box_noise_scale > 0:
        input_query_bbox = add_box_noise(input_query_bbox.clone(), box_noise_scale, negative_gt_mask)
    input_query_class = class_embed(input_query_class)

    # get attention mask
    tgt_size = num_denoising + num_queries
    attn_mask = get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_class, input_query_bbox, attn_mask, dn_meta


def extract_epoch_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    # get epoch id
    epoch_id = int(numbers[0])
    return epoch_id


# get last epoch bias file
def get_bias_file(folder_path):
    # get all txt files
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    # get the latest bias file
    files_name = [os.path.basename(file) for file in txt_files]
    files_epoch = np.array([extract_epoch_from_filename(fn) for fn in files_name])
    max_idx = np.argmax(files_epoch)

    bias_file = txt_files[max_idx]
    return bias_file


def read_txt_to_numpy(bias_filepath):
    with open(bias_filepath, 'r') as file:
        data_list = [np.fromstring(line, dtype=float, sep=' ') for line in file]

    data_array = np.array(data_list)
    return data_array


def calculate_scale(box1, box2, cls=None):
    cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
    cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
    scalex = 2 * (cx2 - cx1) / w1
    scaley = 2 * (cy2 - cy1) / h1
    scalew = (w2 - w1) / w1
    scaleh = (h2 - h1) / h1

    return np.hstack([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)])


def calculate_accuracy(y_true, y_pred):
    """
    calculate accura
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred should keep same length")

    correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
    accuracy = correct_predictions / len(y_true)
    return accuracy


def filter_bias(scales, high_filter: float, low_filter: float, num_noise: int):
    # filter bias
    drop_fined = (np.abs(scales) > high_filter).any(axis=1)
    scales = scales[drop_fined]
    drop_too_large = (np.abs(scales) < low_filter).all(axis=1)
    scales = scales[drop_too_large]
    # get sampling noise (can work when nums_scales < num_noise)
    num_scales = len(scales)
    rand_idx = np.random.randint(0, num_scales, size=num_noise)
    sampling_box_noise = scales[rand_idx]
    return sampling_box_noise


def get_sampling_noise(output_path, num_noise=10000, high_filter=0.1, low_filter=2.0):
    # get last epoch bias
    bias_file = get_bias_file(output_path)
    biases = read_txt_to_numpy(bias_file)
    # filtering boxes
    tbox = biases[:, 1:5]
    pbox = biases[:, 6:]
    scales = calculate_scale(tbox, pbox)
    sampling_box_noise = filter_bias(scales, high_filter, low_filter, num_noise)
    cls_noise = 1 - calculate_accuracy(biases[:, 0], biases[:, 5])
    return sampling_box_noise, cls_noise


# sampling cdn imply
def get_sampling_cdn_training_group(targets,
                                    num_classes,
                                    num_queries,
                                    class_embed,
                                    sampling_box_noise,
                                    label_noise_ratio=0.5,
                                    num_denoising=100,
                                    ):
    """
    Sampling Contrastive Denoising
    """
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None
    num_group = get_num_group(num_denoising, max_gt_num)
    # pad gt to max_num of a batch
    bs = len(num_gts)

    # get contrastive group and mask
    input_query_class, input_query_bbox, pad_gt_mask = init_query_and_mask(bs, max_gt_num, num_classes,
                                                                           num_gts, targets, num_group, device)
    # get mask and idx
    positive_gt_mask, negative_gt_mask, dn_positive_idx = (
        get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device))

    # add noise to box and class
    num_denoising = int(max_gt_num * 2 * num_group)     # total denoising queries
    if label_noise_ratio > 0:
        input_query_class = add_label_noise(input_query_class.clone(), label_noise_ratio, num_classes, pad_gt_mask)

    if sampling_box_noise is not None:
        input_query_bbox = add_box_noise(input_query_bbox.clone(), sampling_box_noise, negative_gt_mask)

    input_query_class = class_embed(input_query_class)
    # get attention mask
    tgt_size = num_denoising + num_queries
    attn_mask = get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_class, input_query_bbox, attn_mask, dn_meta


# An imply only work on single gpu
# 修改一下噪声的参数
class SamplingCDN:
    def __init__(self, num_update=5000, num_noise=10000, high_filter=0.01, low_filter=1.0,
                 positive_box_noise=0.1, negative_box_noise=0.5,  bias_dir=None):
        self.num_update = num_update
        self.num_noise = num_noise
        self.high_filter = high_filter
        self.low_filter = low_filter
        self.positive_noise = positive_box_noise
        self.negative_noise = negative_box_noise
        # dynamic params
        self.sample_count = 0
        self.sampling_box_noise = []
        self.noise_collection = []
        self.label_noise = 0.8
        # init noise
        self.init_sampling(bias_dir)

    def gene_random_xywh_noise_scale(self):
        rand_scales = 2.0 * torch.rand(self.num_noise, 4) - 1.0
        return rand_scales

    def init_sampling(self, bias_dir=None):
        # TODO： solve deal bias file
        if bias_dir:    # load coco_bias_dir
            assert 1, "can not deal bias_file now"
            pass
        else:   # use random noise like cdn
            self.sampling_box_noise = self.gene_random_xywh_noise_scale()

    def update_sampling_noise(self):
        # deal new noise
        self.sampling_box_noise = torch.cat(self.noise_collection.copy(), dim=0)
        self.sample_count = 0
        self.noise_collection = []
        # TODO: update box noise and label noise
        self.sampling_box_noise, self.label_noise = self.filter_sampling_noise()

    def save_model_noise(self, noises, batch_size):
        self.sample_count += batch_size
        self.noise_collection.append(noises.to('cpu'))
        # update noise if counts >= num_update
        if self.sample_count >= self.num_update:
            # print("updating noise")
            self.update_sampling_noise()

    def filter_bias(self, scales, high_filter: float, low_filter: float, num_noise: int):
        # filter bias
        drop_fined = torch.abs(scales) > high_filter
        drop_fined = torch.any(drop_fined, dim=1)
        scales = scales[drop_fined]

        drop_poor = torch.abs(scales) < low_filter
        drop_poor = torch.all(drop_poor, dim=1)
        scales = scales[drop_poor]
        # get sampling noise (can work when nums_scales < num_noise)
        num_scales = len(scales)
        rand_idx = np.random.randint(0, num_scales, size=num_noise)
        sampling_box_noise = scales[rand_idx]
        return sampling_box_noise

    def calculate_scale(self, box1, box2):
        cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
        cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
        scalex = 2 * (cx2 - cx1) / w1
        scaley = 2 * (cy2 - cy1) / h1
        scalew = (w2 - w1) / w1
        scaleh = (h2 - h1) / h1

        return torch.cat([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)],
                         dim=1)

    def calculate_accuracy(self, y_true, y_pred):
        """
        calculate accura
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred should keep same length")

        correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def filter_sampling_noise(self):
        # filtering boxes
        # self.sampling_box_noise = torch.tensor(self.sampling_box_noise)
        tbox = self.sampling_box_noise[:, 1:5]
        pbox = self.sampling_box_noise[:, 6:]
        scales = self.calculate_scale(tbox, pbox)
        bbox_noise = self.filter_bias(scales, self.high_filter, self.low_filter, self.num_noise)
        cls_noise = 1 - self.calculate_accuracy(self.sampling_box_noise[:, 0], self.sampling_box_noise[:, 5])
        return bbox_noise, cls_noise

    def init_query_and_mask(self, bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
        input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
        input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
        pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
        for i in range(bs):
            num_gt = num_gts[i]
            if num_gt > 0:
                input_query_class[i, :num_gt] = targets[i]['labels']
                input_query_bbox[i, :num_gt] = targets[i]['boxes']
                pad_gt_mask[i, :num_gt] = 1
        # each group has positive and negative queries.
        input_query_class = input_query_class.tile([1, 2 * num_group])
        input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
        pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
        return input_query_class, input_query_bbox, pad_gt_mask

    def solve_error_labels(self, noise_mask, max_gt_num, pad_gt_mask):
        # only save useful mask
        noise_mask = noise_mask & pad_gt_mask
        split_noise_mask = torch.split(noise_mask, max_gt_num, dim=1)
        positive_mask = split_noise_mask[0::2]
        negative_mask = split_noise_mask[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_mask, ng_mask) in enumerate(zip(positive_mask, negative_mask)):
            label_errors = pos_mask > ng_mask  # error when pos_iou > ng_iou(True > False)
            # label_errors = label_errors.to(torch.bool)
            image_id, idx = torch.where(label_errors)     # it is a torch.bool tensor
            if idx.numel() != 0:
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组,swap一下
        for idx in error_idx:
            pos_box = noise_mask[idx[0], idx[1]].clone()
            ng_box = noise_mask[idx[0], idx[1] + max_gt_num].clone()
            noise_mask[idx[0], idx[1]], noise_mask[idx[0], idx[1] + max_gt_num] = ng_box, pos_box

        return noise_mask

    # one 2 one box iou
    def batch_box_iou(self, gt_box, noised_box):
        res_iou = []
        for gt, noised in zip(gt_box, noised_box):
            res_iou.append(matched_box_iou(gt, noised).unsqueeze(0))
        return torch.cat(res_iou, dim=0)

    def solve_error_bbox(self, gt_box, noised_box, max_gt_num, pad_gt_mask):
        noised_iou = self.batch_box_iou(gt_box, noised_box)
        # 去掉无关的
        noised_iou = noised_iou.masked_fill(~pad_gt_mask, 0.0)
        split_iou = torch.split(noised_iou, max_gt_num, dim=1)
        positive_iou = split_iou[0::2]
        negative_iou = split_iou[1::2]
        error_idx = []
        # 判断每个组是否有问题
        for i, (pos_iou, ng_iou) in enumerate(zip(positive_iou, negative_iou)):
            box_errors = pos_iou < ng_iou  # error when pos_iou < ng_iou
            # box_errors = torch.tensor(box_errors, dtype=torch.bool)
            image_id, idx = torch.where(box_errors)
            if idx.numel() != 0:
                # print('error box appear')
                idx += i * (max_gt_num * 2)  # get origin idx
                box_error_idx = list(zip(image_id, idx))
                error_idx.extend(box_error_idx)
        # 解决问题组,swap一下
        for idx in error_idx:
            pos_box = noised_box[idx[0], idx[1]].clone()
            ng_box = noised_box[idx[0], idx[1] + max_gt_num].clone()
            noised_box[idx[0], idx[1]], noised_box[idx[0], idx[1] + max_gt_num] = ng_box, pos_box
        return noised_box

    def add_label_noise(self, input_query_class, max_gt_num, num_classes, pad_gt_mask):
        noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.label_noise
        # todo: how to make sure negative box is worse than positive box 副样本有点太差了，需要更好的样本
        # 如果发现positive组有噪声，就和negative组交换
        noise_mask = self.solve_error_labels(noise_mask, max_gt_num, pad_gt_mask)
        # randomly put a new one here
        new_label = torch.randint_like(noise_mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(noise_mask, new_label, input_query_class)

        return input_query_class

    # TODO: use iou_based sampling to get hard samples
    def add_hard_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            noise_scale.append(tmp_noise.unsqueeze(0))
        noise_scale = torch.cat(noise_scale, dim=0).to(input_query_bbox.device)
        # positive_noise_scale = torch.cat([torch.tensor(self.sampling_box_noise[idx]).unsqueeze(0)
        #                                   for idx in batch_noise_idx], dim=0)
        # todo: how to make sure negative box is worse than positive box
        # add random positive and negative noise, this negative is not a good method, need iou
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0     # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = torch.rand_like(input_query_bbox) * self.negative_noise + self.positive_noise    # 有问题em需要让保持nage>positive
        positive_rand_part *= rand_sign
        negative_rand_part *= rand_sign
        positive_noise_scale = noise_scale + positive_rand_part
        negative_noise_scale = torch.roll(noise_scale, shifts=max_gt_num, dims=1) + negative_rand_part  # add to same noise
        # get diff: diff is wh/2, wh
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_noise_scale * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def add_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
        # get positive noise from model's true noise
        bs = len(input_query_bbox)
        batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
        positive_noise_scale = []
        for idx in batch_noise_idx:
            tmp_noise = self.sampling_box_noise[idx].clone()
            positive_noise_scale.append(tmp_noise.unsqueeze(0))
        positive_noise_scale = torch.cat(positive_noise_scale, dim=0).to(input_query_bbox.device)
        # positive_noise_scale = torch.cat([torch.tensor(self.sampling_box_noise[idx]).unsqueeze(0)
        #                                   for idx in batch_noise_idx], dim=0)
        # todo: how to make sure negative box is worse than positive box
        # add random positive and negative noise, this negative is not a good method, need iou
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0     # + or - noise
        positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
        negative_rand_part = (torch.rand_like(input_query_bbox) + 1.0) * negative_gt_mask
        positive_rand_part *= rand_sign
        negative_rand_part *= rand_sign
        positive_noise_scale += positive_rand_part
        # get diff: diff is w / 2 or h / 2
        # diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2])
        diff = torch.cat((input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]), dim=2)
        positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
        negative_query_bbox = input_query_bbox.clone() + negative_rand_part * diff
        gt_query_box = input_query_bbox.clone()
        # add noise to known bbox
        input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
        # input_query_bbox = torch.where(negative_gt_mask, negative_known_bbox, input_query_bbox)
        # clip box
        noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        noise_bbox.clip_(min=0.0, max=1.0)
        gt_box = box_cxcywh_to_xyxy(gt_query_box)
        # todo: make sure negative box is worse than positive box
        noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
        input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        return input_query_bbox

    def get_num_group(self, num_denoising, max_gt_num):
        num_group = num_denoising // max_gt_num
        num_group = 1 if num_group == 0 else num_group
        return num_group

    def get_contrastive_mask_and_idx(self, bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
        # positive and negative mask
        negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
        negative_gt_mask[:, max_gt_num:] = 1
        negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
        positive_gt_mask = 1 - negative_gt_mask
        # contrastive denoising training positive index
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        return positive_gt_mask, negative_gt_mask, dn_positive_idx

    def get_attention_mask(self, tgt_size, num_denoising, num_group, max_gt_num, device):
        # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True

        # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
            else:
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
                attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        return attn_mask

    def get_sampling_cdn_group(self, targets, num_classes, num_queries, class_embed, num_denoising=100):
        """
        Get Sampling Contrastive Denoising
            sampling_box_noise,
            label_noise_ratio=0.5,
        """
        # step0: init params
        if num_denoising <= 0:
            return None, None, None, None

        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        max_gt_num = max(num_gts)
        if max_gt_num == 0:
            return None, None, None, None
        num_group = self.get_num_group(num_denoising, max_gt_num)
        # pad gt to max_num of a batch
        bs = len(num_gts)

        # step1: get contrastive group and mask
        input_query_class, input_query_bbox, pad_gt_mask = self.init_query_and_mask(bs, max_gt_num, num_classes,
                                                                               num_gts, targets, num_group, device)
        # get mask and positive idx
        positive_gt_mask, negative_gt_mask, dn_positive_idx = (
            self.get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device))

        # step3: add noise to box and class
        num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries
        if self.label_noise > 0:
            input_query_class = self.add_label_noise(input_query_class, max_gt_num, num_classes, pad_gt_mask)

        if self.sampling_box_noise is not None:
            input_query_bbox = self.add_hard_box_noise(input_query_bbox, num_denoising, negative_gt_mask,
                                                  max_gt_num, pad_gt_mask)

        input_query_class = class_embed(input_query_class)
        # step4: get attention mask
        tgt_size = num_denoising + num_queries
        attn_mask = self.get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)

        dn_meta = {
            "dn_positive_idx": dn_positive_idx,
            "dn_num_group": num_group,
            "dn_num_split": [num_denoising, num_queries]
        }

        return input_query_class, input_query_bbox, attn_mask, dn_meta


# 使用分层采样： An imply only work on single gpu

# class SamplingIoUCDN:
#     def __init__(self, num_update=10000, num_noise=10000, high_filter=0.05, low_filter=1.0,
#                  positive_box_noise=0.05, negative_box_noise=0.1,  bias_dir=None):
#         self.num_update = num_update
#         self.num_noise = num_noise
#         self.high_filter = high_filter
#         self.low_filter = low_filter
#         self.positive_noise = positive_box_noise
#         self.negative_noise = negative_box_noise
#         # dynamic params
#         self.sample_count = 0
#         # self.
#         self.sampling_box_noise = []
#         self.noise_collection = []
#         self.label_noise = 0.25
#         # init noise
#         self.init_sampling(bias_dir)
#
#     def gene_random_xywh_noise_scale(self):
#         rand_scales = 2.0 * torch.rand(self.num_noise, 4) - 1.0
#         return rand_scales
#
#     def init_sampling(self, bias_dir=None):
#         # TODO： solve deal bias file
#         if bias_dir:    # load coco_bias_dir
#             assert 1, "can not deal bias_file now"
#             pass
#         else:   # use random noise like cdn
#             self.sampling_box_noise = self.gene_random_xywh_noise_scale()
#
#     def update_sampling_noise(self):
#         # deal new noise
#         self.sampling_box_noise = torch.cat(self.noise_collection.copy(), dim=0)
#         self.sample_count = 0
#         self.noise_collection = []
#         # TODO: update box noise and label noise
#         self.sampling_box_noise, self.label_noise = self.filter_sampling_noise()
#
#     def save_model_noise(self, noises, batch_size):
#         self.sample_count += batch_size
#         self.noise_collection.append(noises.to('cpu'))
#         # update noise if counts >= num_update
#         if self.sample_count >= self.num_update:
#             # print("updating noise")
#             self.update_sampling_noise()
#
#     def filter_bias(self, scales, high_filter: float, low_filter: float, num_noise: int):
#         # filter bias
#         drop_fined = torch.abs(scales) > high_filter
#         drop_fined = torch.any(drop_fined, dim=1)
#         scales = scales[drop_fined]
#
#         drop_poor = torch.abs(scales) < low_filter
#         drop_poor = torch.all(drop_poor, dim=1)
#         scales = scales[drop_poor]
#         # get sampling noise (can work when nums_scales < num_noise)
#         num_scales = len(scales)
#         rand_idx = np.random.randint(0, num_scales, size=num_noise)
#         sampling_box_noise = scales[rand_idx]
#         return sampling_box_noise
#
#     def calculate_scale(self, box1, box2):
#         cx1, cy1, w1, h1 = [box1[:, i] for i in range(box1.shape[1])]
#         cx2, cy2, w2, h2 = [box2[:, i] for i in range(box2.shape[1])]
#         scalex = 2 * (cx2 - cx1) / w1
#         scaley = 2 * (cy2 - cy1) / h1
#         scalew = (w2 - w1) / w1
#         scaleh = (h2 - h1) / h1
#
#         return torch.cat([scalex.reshape(-1, 1), scaley.reshape(-1, 1), scalew.reshape(-1, 1), scaleh.reshape(-1, 1)],
#                          dim=1)
#
#     def calculate_accuracy(self, y_true, y_pred):
#         """
#         calculate accura
#         """
#         if len(y_true) != len(y_pred):
#             raise ValueError("y_true and y_pred should keep same length")
#
#         correct_predictions = sum(int(yt) == int(yp) for yt, yp in zip(y_true, y_pred))
#         accuracy = correct_predictions / len(y_true)
#         return accuracy
#
#     def filter_sampling_noise(self):
#         # filtering boxes
#         # self.sampling_box_noise = torch.tensor(self.sampling_box_noise)
#         tbox = self.sampling_box_noise[:, 1:5]
#         pbox = self.sampling_box_noise[:, 6:]
#         scales = self.calculate_scale(tbox, pbox)
#         bbox_noise = self.filter_bias(scales, self.high_filter, self.low_filter, self.num_noise)
#         cls_noise = 1 - self.calculate_accuracy(self.sampling_box_noise[:, 0], self.sampling_box_noise[:, 5])
#         return bbox_noise, cls_noise
#
#     def init_query_and_mask(self, bs, max_gt_num, num_classes, num_gts, targets, num_group, device):
#         input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
#         input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
#         pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
#         for i in range(bs):
#             num_gt = num_gts[i]
#             if num_gt > 0:
#                 input_query_class[i, :num_gt] = targets[i]['labels']
#                 input_query_bbox[i, :num_gt] = targets[i]['boxes']
#                 pad_gt_mask[i, :num_gt] = 1
#         # each group has positive and negative queries.
#         input_query_class = input_query_class.tile([1, 2 * num_group])
#         input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
#         pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
#         return input_query_class, input_query_bbox, pad_gt_mask
#
#     def solve_error_labels(self, noise_mask, max_gt_num, pad_gt_mask):
#         # only save useful mask
#         noise_mask = noise_mask & pad_gt_mask
#         split_noise_mask = torch.split(noise_mask, max_gt_num, dim=1)
#         positive_mask = split_noise_mask[0::2]
#         negative_mask = split_noise_mask[1::2]
#         error_idx = []
#         # 判断每个组是否有问题
#         for i, (pos_mask, ng_mask) in enumerate(zip(positive_mask, negative_mask)):
#             label_errors = pos_mask > ng_mask  # error when pos_iou > ng_iou(True > False)
#             # label_errors = label_errors.to(torch.bool)
#             image_id, idx = torch.where(label_errors)     # it is a torch.bool tensor
#             if idx.numel() != 0:
#                 idx += i * (max_gt_num * 2)  # get origin idx
#                 box_error_idx = list(zip(image_id, idx))
#                 error_idx.extend(box_error_idx)
#         # 解决问题组,swap一下
#         for idx in error_idx:
#             pos_box = noise_mask[idx[0], idx[1]].clone()
#             ng_box = noise_mask[idx[0], idx[1] + max_gt_num].clone()
#             noise_mask[idx[0], idx[1]], noise_mask[idx[0], idx[1] + max_gt_num] = ng_box, pos_box
#
#         return noise_mask
#
#     # one 2 one box iou
#     def batch_box_iou(self, gt_box, noised_box):
#         res_iou = []
#         for gt, noised in zip(gt_box, noised_box):
#             res_iou.append(matched_box_iou(gt, noised).unsqueeze(0))
#         return torch.cat(res_iou, dim=0)
#
#     def solve_error_bbox(self, gt_box, noised_box, max_gt_num, pad_gt_mask):
#         noised_iou = self.batch_box_iou(gt_box, noised_box)
#         # 去掉无关的
#         noised_iou = noised_iou.masked_fill(~pad_gt_mask, 0.0)
#         split_iou = torch.split(noised_iou, max_gt_num, dim=1)
#         positive_iou = split_iou[0::2]
#         negative_iou = split_iou[1::2]
#         error_idx = []
#         # 判断每个组是否有问题
#         for i, (pos_iou, ng_iou) in enumerate(zip(positive_iou, negative_iou)):
#             box_errors = pos_iou < ng_iou  # error when pos_iou < ng_iou
#             # box_errors = torch.tensor(box_errors, dtype=torch.bool)
#             image_id, idx = torch.where(box_errors)
#             if idx.numel() != 0:
#                 # print('error box appear')
#                 idx += i * (max_gt_num * 2)  # get origin idx
#                 box_error_idx = list(zip(image_id, idx))
#                 error_idx.extend(box_error_idx)
#         # 解决问题组,swap一下
#         for idx in error_idx:
#             pos_box = noised_box[idx[0], idx[1]].clone()
#             ng_box = noised_box[idx[0], idx[1] + max_gt_num].clone()
#             noised_box[idx[0], idx[1]], noised_box[idx[0], idx[1] + max_gt_num] = ng_box, pos_box
#         return noised_box
#
#     def add_label_noise(self, input_query_class, max_gt_num, num_classes, pad_gt_mask):
#         noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < self.label_noise
#         # todo: how to make sure negative box is worse than positive box 副样本有点太差了，需要更好的样本
#         # 如果发现positive组有噪声，就和negative组交换
#         noise_mask = self.solve_error_labels(noise_mask, max_gt_num, pad_gt_mask)
#         # randomly put a new one here
#         new_label = torch.randint_like(noise_mask, 0, num_classes, dtype=input_query_class.dtype)
#         input_query_class = torch.where(noise_mask, new_label, input_query_class)
#
#         return input_query_class
#
#     # TODO: use iou_based sampling to get hard samples
#     def add_box_noise(self, input_query_bbox, num_denoising, negative_gt_mask, max_gt_num, pad_gt_mask):
#         # get positive noise from model's true noise
#         bs = len(input_query_bbox)
#         batch_noise_idx = [torch.randint(0, self.num_noise, size=(num_denoising,)) for _ in range(bs)]
#         positive_noise_scale = []
#         for idx in batch_noise_idx:
#             tmp_noise = self.sampling_box_noise[idx].clone()
#             positive_noise_scale.append(tmp_noise.unsqueeze(0))
#         positive_noise_scale = torch.cat(positive_noise_scale, dim=0).to(input_query_bbox.device)
#         # positive_noise_scale = torch.cat([torch.tensor(self.sampling_box_noise[idx]).unsqueeze(0)
#         #                                   for idx in batch_noise_idx], dim=0)
#         # todo: how to make sure negative box is worse than positive box
#         # add random positive and negative noise, this negative is not a good method, need iou
#         rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0     # + or - noise
#         positive_rand_part = torch.rand_like(input_query_bbox) * self.positive_noise
#         negative_rand_part = (torch.rand_like(input_query_bbox) + 1.0) * negative_gt_mask
#         positive_rand_part *= rand_sign
#         negative_rand_part *= rand_sign
#         positive_noise_scale += positive_rand_part
#         # get diff: diff is w / 2 or h / 2
#         diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2])
#         positive_query_bbox = input_query_bbox.clone() + positive_noise_scale * diff
#         negative_query_bbox = input_query_bbox.clone() + negative_rand_part * diff
#         gt_query_box = input_query_bbox.clone()
#         # add noise to known bbox
#         input_query_bbox = torch.where(negative_gt_mask == 1.0, negative_query_bbox, positive_query_bbox)
#         # input_query_bbox = torch.where(negative_gt_mask, negative_known_bbox, input_query_bbox)
#         # clip box
#         noise_bbox = box_cxcywh_to_xyxy(input_query_bbox)
#         noise_bbox.clip_(min=0.0, max=1.0)
#         gt_box = box_cxcywh_to_xyxy(gt_query_box)
#         # todo: make sure negative box is worse than positive box
#         noise_bbox = self.solve_error_bbox(gt_box, noise_bbox, max_gt_num, pad_gt_mask)
#         input_query_bbox = box_xyxy_to_cxcywh(noise_bbox)
#         input_query_bbox = inverse_sigmoid(input_query_bbox)
#         return input_query_bbox
#
#     def get_num_group(self, num_denoising, max_gt_num):
#         num_group = num_denoising // max_gt_num
#         num_group = 1 if num_group == 0 else num_group
#         return num_group
#
#     def get_contrastive_mask_and_idx(self, bs, num_gts, max_gt_num, num_group, pad_gt_mask, device):
#         # positive and negative mask
#         negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
#         negative_gt_mask[:, max_gt_num:] = 1
#         negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
#         positive_gt_mask = 1 - negative_gt_mask
#         # contrastive denoising training positive index
#         positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
#         dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
#         dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
#         return positive_gt_mask, negative_gt_mask, dn_positive_idx
#
#     def get_attention_mask(self, tgt_size, num_denoising, num_group, max_gt_num, device):
#         # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
#         attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
#         # match query cannot see the reconstruction
#         attn_mask[num_denoising:, :num_denoising] = True
#
#         # reconstruct cannot see each other
#         for i in range(num_group):
#             if i == 0:
#                 attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
#             if i == num_group - 1:
#                 attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
#             else:
#                 attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
#                 attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
#         return attn_mask
#
#     def get_sampling_cdn_group(self, targets, num_classes, num_queries, class_embed, num_denoising=100):
#         """
#         Get Sampling Contrastive Denoising
#             sampling_box_noise,
#             label_noise_ratio=0.5,
#         """
#         # step0: init params
#         if num_denoising <= 0:
#             return None, None, None, None
#
#         num_gts = [len(t['labels']) for t in targets]
#         device = targets[0]['labels'].device
#         max_gt_num = max(num_gts)
#         if max_gt_num == 0:
#             return None, None, None, None
#         num_group = self.get_num_group(num_denoising, max_gt_num)
#         # pad gt to max_num of a batch
#         bs = len(num_gts)
#
#         # step1: get contrastive group and mask
#         input_query_class, input_query_bbox, pad_gt_mask = self.init_query_and_mask(bs, max_gt_num, num_classes,
#                                                                                num_gts, targets, num_group, device)
#         # get mask and positive idx
#         positive_gt_mask, negative_gt_mask, dn_positive_idx = (
#             self.get_contrastive_mask_and_idx(bs, num_gts, max_gt_num, num_group, pad_gt_mask, device))
#
#         # step3: add noise to box and class
#         num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries
#         if self.label_noise > 0:
#             input_query_class = self.add_label_noise(input_query_class, max_gt_num, num_classes, pad_gt_mask)
#
#         if self.sampling_box_noise is not None:
#             input_query_bbox = self.add_box_noise(input_query_bbox, num_denoising, negative_gt_mask,
#                                                   max_gt_num, pad_gt_mask)
#
#         input_query_class = class_embed(input_query_class)
#         # step4: get attention mask
#         tgt_size = num_denoising + num_queries
#         attn_mask = self.get_attention_mask(tgt_size, num_denoising, num_group, max_gt_num, device)
#
#         dn_meta = {
#             "dn_positive_idx": dn_positive_idx,
#             "dn_num_group": num_group,
#             "dn_num_split": [num_denoising, num_queries]
#         }
#
#         return input_query_class, input_query_bbox, attn_mask, dn_meta




# the lock may not fit our need
# class SamplingCDN:
#     def __init__(self, num_update=10000, num_noise=10000, high_filter=0.1, low_filter=2.0):
#         self.num_update = num_update
#         self.num_noise = num_noise
#         self.high_filter = high_filter
#         self.low_filter = low_filter
#         # use lock to safely update list
#         self.update_lock = threading.Lock()
#         self.collection_lock = threading.Lock()
#         self.sample_count = 0
#         self.sampling_box_noise = []
#         self.noise_collection = []
#         self.label_noise = 0.25
#
#     def update_sampling_noise(self):
#         with self.update_lock:
#             self.sampling_box_noise = self.noise_collection.copy()
#
#         self.sample_count = 0
#         self.noise_collection = []
#
#     def save_model_noise(self, noises, batch_size):
#         with self.collection_lock:
#             self.sample_count += batch_size
#             self.noise_collection.append(noises)
#             # update noise if counts >= num_update
#             if self.sample_count >= self.num_update:
#                 self.update_sampling_noise()










