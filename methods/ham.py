import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class FeatureExtractor():
    def __init__(self, model, intermediate_layers):
        self.model = model
        self.intermediate_layers = intermediate_layers[::-1]
        self.weights = []
        self.num = len(self.intermediate_layers)
        self.activations = []

    def save_weight(self, grad):
        self.weights.append(grad)

    def __call__(self, x, intermediate_layers, class_idx):
        hams = []
        logit, conv3, conv4, conv5 = self.model(x)
        logit = F.softmax(logit)
        score = logit[:, class_idx].squeeze()
        if torch.cuda.is_available():
          score = score.cuda()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        saliency_map = torch.mul(conv3, self.model.grads['conv3']).mean(dim=1, keepdim=True)
        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        hams.append(F.relu(norm_saliency_map, inplace=True))

        saliency_map = torch.mul(conv4, self.model.grads['conv4']).mean(dim=1, keepdim=True)
        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        hams.append(F.relu(norm_saliency_map, inplace=True))

        saliency_map = torch.mul(conv5, self.model.grads['conv5']).mean(dim=1, keepdim=True)
        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        hams.append(F.relu(norm_saliency_map, inplace=True))


        return hams


class HAM():
    def __init__(self, model, intermediate_layers):
        self.model = model
        self.intermediate_layers = intermediate_layers
        self.extractor = FeatureExtractor(self.model, intermediate_layers)

    def __call__(self, input, device, intermediate_layers, class_idx):
        hams = self.extractor(input, intermediate_layers, class_idx)
        for i in range(len(intermediate_layers)):
            if i == 0:
                aggregated_ham = hams[i]
            else:
                tmp = F.interpolate(hams[i], hams[0].shape[-3:], mode='trilinear', align_corners=False)
                aggregated_ham = torch.mul(tmp, torch.ge(tmp, aggregated_ham)) + torch.mul(aggregated_ham, torch.ge(aggregated_ham, tmp))

        B, L, C, H, W = aggregated_ham.shape
        aggregated_ham = aggregated_ham.view(B, -1)
        aggregated_ham -= aggregated_ham.min(dim=1, keepdim=True)[0]
        aggregated_ham /= aggregated_ham.max(dim=1, keepdim=True)[0]
        aggregated_ham = aggregated_ham.view(B, L, C, H, W)
        aggregated_ham = F.interpolate(aggregated_ham, input.shape[-3:], mode='trilinear', align_corners=False)
        return aggregated_ham