
import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn

class BaselineModel(nn.Module):

    def __init__(self,
                 num_classes: int,
                 args: argparse.Namespace,
                 depth=12):
        super().__init__()
        assert num_classes > 0

        self._num_classes = num_classes

        self.num_channels = args.num_channels
        self.num_features = args.num_features

        self.in_channels = 1
        self.grads = {}
        self.depth = depth

        filters = [50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        for i in range(depth):
            if i == 0:
                block = nn.Conv3d(self.in_channels, filters[i], kernel_size=(5, 5, 5))
            else:
                block = nn.Conv3d(filters[i], filters[i + 1], kernel_size=(3, 3, 3))

            setattr(self, f"block{i + 1}", block)

        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.num_features, num_classes)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def feature_require_grad(self) -> bool:
        return self.feature_layer.feature_vectors.requires_grad

    @feature_require_grad.setter
    def feature_require_grad(self, val: bool):
        self.feature_layer.feature_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def save_grad(self, module, grad_input, grad_output):
        self.grads[module.name] = grad_output[0]

    def forward(self,
                x: torch.Tensor,
                **kwargs,
                ) -> tuple:
        B, _, L, H, W = x.shape
        for i in range(self.depth):
            block = getattr(self, f"block{i + 1}")
            x = block(x)

        x = self.relu(x)
        pred = self.classifier(x.view(B, -1))

        return pred


    def save(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)

