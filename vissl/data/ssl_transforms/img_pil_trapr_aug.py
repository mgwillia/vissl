# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence
from PIL.ImageFilter import GaussianBlur
import random

import numpy as np
import torchvision.transforms as pth_transforms
from torchvision.transforms.functional import gaussian_blur
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image
import torch.nn.functional as F
import torch

import logging

@register_transform("ImgPilTraPrAugment")
class ImgPilPairAugment(ClassyTransform):
    """
    Convert a PIL image to a weak and strong transformed image.
    """

    def __init__(
        self,
        color_distortion_strength: float,
        p_gaussian_blur: float,
        gaussian_radius_min: float,
        gaussian_radius_max: float
    ):

        self.p_gaussian_blur = p_gaussian_blur
        self.gaussian_radius_min = gaussian_radius_min
        self.gaussian_radius_max = gaussian_radius_max
        self.strength = color_distortion_strength

    def __call__(self, image: Image.Image) -> List:

        logging.info(type(image))

        should_color_jitter = np.random.rand() <= 0.8
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            torch.Tensor([-1, -1, -1, -1]), \
            1.0, 1.0, 1.0, 0.0
        if should_color_jitter:
            fn_idx = torch.randperm(4)
            b = float(torch.empty(1).uniform_(1.0 - 0.8 * self.strength, 1.0 + 0.8 * self.strength))
            c = float(torch.empty(1).uniform_(1.0 - 0.8 * self.strength, 1.0 + 0.8 * self.strength))
            s = float(torch.empty(1).uniform_(1.0 - 0.8 * self.strength, 1.0 + 0.8 * self.strength))
            h = float(torch.empty(1).uniform_(0.0 - 0.2 * self.strength, 0.0 + 0.2 * self.strength))

            brightness_factor, contrast_factor, saturation_factor, hue_factor = b, c, s, h

            for fn_id in fn_idx:
                if fn_id == 0:
                    image = pth_transforms.functional.adjust_brightness(image, brightness_factor)
                elif fn_id == 1:
                    image = pth_transforms.functional.adjust_contrast(image, contrast_factor)
                elif fn_id == 2:
                    image = pth_transforms.functional.adjust_saturation(image, saturation_factor)
                elif fn_id == 3:
                    image = pth_transforms.functional.adjust_hue(image, hue_factor)

        logging.info(type(image))

        should_grayscale = np.random.rand() <= 0.2
        grayscale = 0.0
        if should_grayscale:
            grayscale = 1.0
            image = pth_transforms.RandomGrayscale(p=1.0)(image)

        logging.info(type(image))

        should_blur = np.random.rand() <= self.p_gaussian_blur
        gaussian_radius = 0.0
        if should_blur:
            gaussian_radius = random.uniform(self.gaussian_radius_min, self.gaussian_radius_max)
            image = image.filter(
                GaussianBlur(
                    radius=gaussian_radius
                )
            )

        transforms = torch.Tensor([
            fn_idx[0] + 1.0, brightness_factor,
            fn_idx[1] + 1.0, contrast_factor,
            fn_idx[2] + 1.0, saturation_factor,
            fn_idx[3] + 1.0, hue_factor,
            grayscale,
            gaussian_radius
        ])

        return image, transforms

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilPairAugment":
        """
        Instantiates ImgPilPairAugment from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilPairAugment instance.
        """
        return cls(**config)
