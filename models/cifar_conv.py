# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """Conv-2 / Conv-4 / Conv-6 networks for CIFAR-10 (DLT-style)."""

    class ConvBlock(nn.Module):
        """(Conv3x3 -> ReLU) x2 + MaxPool(2)."""

        def __init__(self, in_ch, c1, c2):
            super(Model.ConvBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return x

    def __init__(self, depth, initializer, outputs=10):
        super(Model, self).__init__()

        # depth = total number of conv layers (2 / 4 / 6)
        if depth == 2:
            blocks = [(3, 64, 64)]
            flat_dim = 64 * 16 * 16
        elif depth == 4:
            blocks = [(3, 64, 64), (64, 128, 128)]
            flat_dim = 128 * 8 * 8
        elif depth == 6:
            blocks = [(3, 64, 64), (64, 128, 128), (128, 256, 256)]
            flat_dim = 256 * 4 * 4
        else:
            raise ValueError("depth must be one of {2,4,6}")

        layers = []
        for (in_ch, c1, c2) in blocks:
            layers.append(Model.ConvBlock(in_ch, c1, c2))
        self.layers = nn.Sequential(*layers)

        # Fully connected head (matches DLT)
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, outputs)

        self.criterion = nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @property
    def output_layer_names(self):
        return ['fc3.weight', 'fc3.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name in ['cifar_conv_2', 'cifar_conv_4', 'cifar_conv_6']

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10
        depth = int(model_name.split('_')[-1])
        return Model(depth, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        # DEFAULT = Conv2 (as requested)
        model_hparams = hparams.ModelHparams(
            model_name='cifar_conv_2',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=60,
            do_not_augment=False
        )

        # Conv2 hyperparameters from deconstructing-lottery-tickets
        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=2e-4,
            weight_decay=0.0,
            training_steps='27ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc3.weight'
        )

        return LotteryDesc(
            model_hparams,
            dataset_hparams,
            training_hparams,
            pruning_hparams
        )
