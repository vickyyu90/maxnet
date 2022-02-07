
import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn

from util.func import min_pool3d
import torch.nn.functional as F
from .utils import UnetConv3, PadMaxPool3d
from .dual_attention import DualAttentionBlock

from .l2dist import L2Conv3D

class MAXNet(nn.Module):

    def __init__(self,
                 num_classes: int,
                 args: argparse.Namespace
                 ):
        super().__init__()
        assert num_classes > 0

        self._num_classes = num_classes

        self.num_channels = args.num_channels
        self.num_features = args.num_features
        self.feature_shape = (args.L1, args.W1, args.H1, args.num_channels)

        self.in_channels = 1
        self.feature_scale = 4
        self.is_batchnorm = True
        self.grads = {}
        self.epsilon = 1e-4

        filters = [15, 25, 50, 50]

        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3))
        self.max1 = PadMaxPool3d(2, 2)

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3, 3, 3))

        self.max2 = PadMaxPool3d(2, 2)

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3, 3, 3))
        self.conv3.name = 'conv3'
        self.max3 = PadMaxPool3d(2, 2)

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3, 3, 3))
        self.conv4.name = 'conv4'
        self.max4 = PadMaxPool3d(2, 2)

        self.conv5 = UnetConv3(filters[3], filters[3], self.is_batchnorm, kernel_size=(1, 1, 1))
        self.conv5.name = 'conv5'

        # Aggreagation Strategies
        self.combiner1 = DualAttentionBlock(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[2], sub_sample_factor=(2, 2, 2))

        self.combiner2 = DualAttentionBlock(in_channels=filters[3], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(2, 2, 2))

        self.combiner1_scalar1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.combiner1_scalar2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.combiner2_scalar1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.combiner2_scalar2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.classifier = nn.Linear(self.num_features, num_classes)

        self._add_on = nn.Sequential(
                    nn.Conv3d(in_channels=filters[3], out_channels=args.num_channels, kernel_size=1, bias=False),
                    nn.Sigmoid()
                    )

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

        self.feature_layer = L2Conv3D(self.num_features,
                                        self.num_channels,
                                        args.W1,
                                        args.H1,
                                        args.L1)

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
                xs: torch.Tensor,
                **kwargs,
                ) -> tuple:
        conv1 = self.conv1(xs)
        maxpool1 = self.max1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.max2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.max3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.max4(conv4)

        conv5 = self.conv5(maxpool4)

        batch_size = xs.shape[0]
        #pooled = F.adaptive_avg_pool3d(conv5, (1, 1, 1)).view(batch_size, -1)

        # Attention Mechanism
        p1, d1 = self.combiner1(conv3, conv5)
        p2, d2 = self.combiner2(conv4, conv5)

        f3 = self.combiner1_scalar1 * p1 + self.combiner1_scalar2 * d1
        f4 = self.combiner2_scalar1 * p2 + self.combiner2_scalar2 * d2

        aggregated_features = torch.max(F.upsample(conv5, size=f3.size()[2:], mode='trilinear'), f3)
        aggregated_features = torch.max(F.upsample(f4, size=f3.size()[2:], mode='trilinear'), aggregated_features)
        aggregated_features = self._add_on(aggregated_features)
        bs, D, C, W, H = aggregated_features.shape

        # Use the features to compute distances
        distances = self.feature_layer(aggregated_features)
        min_distances = min_pool3d(distances, kernel_size=(C, W, H))
        min_distances = min_distances.view(bs, self.num_features)
        pred = self.classifier(-torch.log(min_distances) + 1e-4)

        if not self._log_probabilities:
            similarities = torch.exp(-min_distances)
        else:
            similarities = -min_distances
        return pred, similarities


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
        # Save the out_map of the model to the target directory
        with open(directory_path + '/tree.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

