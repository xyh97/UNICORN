"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import random
from lavis.common.registry import registry
from lavis.datasets.data_utils import load_video
from lavis.processors import transforms_video
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import VideoRandomAugment
from lavis.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import CLIPImageProcessor

MAX_INT = registry.get("MAX_INT")


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        temp_aug=True,
        pretrain=False,
        mask=False,
        rand_aug=False,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.temp_aug = temp_aug
        self.rand_aug = rand_aug
        self.pretrain = pretrain
        self.mask = mask
        # self.transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if self.rand_aug:
            self.transform = transforms.Compose(
                [
                    # Video size is (C, T, H, W)
                    transforms_video.RandomResizedCropVideo(
                        image_size,
                        scale=(min_scale, max_scale),
                        interpolation_mode="bicubic",
                    ),
                    transforms_video.RandomHorizontalFlipVideo(),
                    ToTHWC(),  # C, T, H, W -> T, H, W, C
                    VideoRandomAugment(
                        2,
                        5,
                        augs=[
                            "Identity",
                            "AutoContrast",
                            "Brightness",
                            "Sharpness",
                            "Equalize",
                            "ShearX",
                            "ShearY",
                            "TranslateX",
                            "TranslateY",
                            "Rotate",
                        ],
                    ),
                    ToUint8(),
                    transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                    self.normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # Video size is (C, T, H, W)
                    transforms_video.RandomResizedCropVideo(
                        image_size,
                        scale=(min_scale, max_scale),
                        interpolation_mode="bicubic",
                    ),
                    # transforms_video.RandomHorizontalFlipVideo(),
                    ToTHWC(),  # C, T, H, W -> T, H, W, C
                    ToUint8(),
                    transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                    self.normalize,
                ]
            )

    def __call__(self, vpath, start, end):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        if self.temp_aug:
            prob = random.random()
        else:
            prob = 1.0
        clip, new_start, new_end, crop = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            sampling="uniform",
            height=self.image_size,
            width=self.image_size,
            s=start,
            e=end,
            prob=prob,
            pretrain=self.pretrain,
            mask=self.mask
        )
        # clip = self.transform.preprocess(clip, return_tensors='pt')['pixel_values']
        # clip = clip.permute(1, 0, 2, 3)
        clip = self.transform(clip)
        return clip, new_start, new_end, crop

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        temp_aug = cfg.get("temp_aug", True)
        rand_aug = cfg.get("rand_aug", False)
        pretrain = cfg.get("pretrain", False)
        mask = cfg.get("mask", False)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            temp_aug=temp_aug,
            rand_aug=rand_aug,
            pretrain=pretrain,
            mask=mask,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, temp_aug=False, mean=None, std=None, n_frms=MAX_INT, mask=False):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.temp_aug = temp_aug
        self.pretrain = False
        self.mask = mask

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )
        # self.transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __call__(self, vpath, start, end):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        if self.temp_aug:
            prob = 0.5
        else:
            prob = 1.0
        if type(start) is list:
            max_len = 0
            max_idx = 0
            for i, (s, e) in enumerate(zip(start, end)):
                if e-s > max_len:
                    max_len = e-s
                    max_idx = i
            start = start[i]
            end = end[i]
        clip, new_start, new_end, crop = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            s=start,
            e=end,
            prob=prob,
            height=self.image_size,
            width=self.image_size,
            mask=self.mask
        )
        # clip = self.transform.preprocess(clip, return_tensors='pt')['pixel_values']
        # clip = clip.permute(1, 0, 2, 3)
        clip = self.transform(clip)

        return clip, new_start, new_end, crop

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        temp_aug = cfg.get("temp_aug", False)
        mask = cfg.get("mask", False)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, temp_aug=temp_aug, mean=mean, std=std, n_frms=n_frms)
