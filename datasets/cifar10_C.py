# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10
from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from datasets.utils.image_corputions import corruption_dict
from utils.conf import base_path


class CIFAR10Corupted(ContinualDataset):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = "cifar10-c"
    SETTING = "domain-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    # N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    TASK_CORRUPTIONS = {
        0: "gaussian_noise",
        1: "shot_noise",
        2: "impulse_noise",
        3: "defocus_blur",
        4: "gaussian_blur",
        5: "motion_blur",
        6: "speckle_noise",
        7: "jpeg_compression",
        8: "pixelate",
        9: "frost",
        10: "snow",
        11: "fog",
        12: "spatter",
        13: "contrast",
        14: "brightness",
        15: "saturate",
        16: "elastic_transform",
        17: "glass_blur",
        18: "zoom_blur",
        19: "clean",
    }

    SEVERITY = 2

    TEST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )

    def __init__(self, args, transform_type: str = "weak"):
        super().__init__(args)

        assert transform_type in ["weak", "strong"], (
            "Transform type must be either 'weak' or 'strong'."
        )

        if transform_type == "strong":
            logging.info("Using strong augmentation for CIFAR10")
            self.TRANSFORM = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10Corupted.MEAN, CIFAR10Corupted.STD),
                ]
            )

            self.SEVERITY = 5

    def get_data_loaders(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        train_dataset = MyCIFAR10(
            base_path() + "CIFAR10", train=True, download=True, transform=transform
        )
        test_dataset = TCIFAR10(
            base_path() + "CIFAR10",
            train=False,
            download=True,
            transform=self.TEST_TRANSFORM,
        )

        corruption = self.TASK_CORRUPTIONS[self.current_task]

        train_dataset.data = corruption_dict[corruption](
            train_dataset.data, self.SEVERITY
        )
        test_dataset.data = corruption_dict[corruption](
            test_dataset.data, self.SEVERITY
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), CIFAR10Corupted.TRANSFORM]
        )
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(CIFAR10Corupted.MEAN, CIFAR10Corupted.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(CIFAR10Corupted.MEAN, CIFAR10Corupted.STD)
        return transform

    @set_default_from_args("n_epochs")
    def get_epochs():
        return 50

    @set_default_from_args("batch_size")
    def get_batch_size():
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR10(base_path() + "CIFAR10", train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
