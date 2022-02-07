import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3,
                 sub_sample_factor=(2,2,2)):
        super(DualAttentionBlock, self).__init__()

        assert dimension in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            bn = nn.BatchNorm3d
        elif dimension == 2:
            bn = nn.BatchNorm2d
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            #conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)

        self.conv2 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x, g):
        '''
        :param x: (B, C, H, W, L)
        :param g: (B, C, H, W, L)
        :return:
        '''
        # F_n
        theta_x = self.conv1(x)
        x_size = theta_x.size()

        # G_A
        theta_g = self.conv2(g)
        theta_g = F.upsample(theta_g, size=x_size[2:], mode='trilinear')


        theta_c = torch.mul(theta_x, theta_g)
        theta_c = theta_c - torch.min(theta_c)
        theta_c = theta_c/torch.sum(theta_c, dim=(2, 3, 4), keepdim=True)

        p = torch.mul(theta_c, x)

        # depth-wise
        theta_x_d = theta_x.permute(0, 2, 3, 4, 1)
        theta_g_d = theta_g.permute(0, 2, 3, 4, 1)
        theta_c_d = theta_x_d.permute(0, 1, 2, 4, 3) @ (theta_g_d)
        theta_c_d = theta_c_d - torch.min(theta_c_d)
        theta_c_d = theta_c_d/torch.sum(theta_c_d, dim=(1, 2, 3, 4), keepdim=True)
        d = (x.permute(0, 2, 3, 4, 1) @ theta_c_d).permute(0, 4, 1, 2, 3)

        return p, d


