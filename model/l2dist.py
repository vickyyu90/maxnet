
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Conv3D(nn.Module):
    def __init__(self, num_features, num_channels, w_1, h_1, L_1):
        super().__init__()
        feature_shape = (num_features, num_channels, w_1, h_1, L_1)
        self.feature_vectors = nn.Parameter(torch.randn(feature_shape), requires_grad=True)

    def forward(self, xs):
        ones = torch.ones_like(self.feature_vectors,
                               device=xs.device)
        xs_squared_l2 = F.conv3d(xs ** 2, weight=ones)
        # squared L2 distance
        ps_squared_l2 = torch.sum(self.feature_vectors ** 2, dim=(1, 2, 3, 4))
        # Reshape the tensor
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1, 1)

        xs_conv = F.conv3d(xs, weight=self.feature_vectors)

        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        distance = torch.sqrt(torch.abs(distance)+1e-14)

        return distance
