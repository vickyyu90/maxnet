from model.model import MAXNet
from model.BaseModel import BaselineModel
from util.data import get_dataloaders
from evaluation import CausalMetric, auc
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from shutil import copy
from copy import deepcopy
import cv2
import os
from util.args import get_args
from explanations import RISE
import nibabel as nib
from util.data import FILENAME_TYPE
from util.data import MinMaxNormalization
from methods.scorecam import ScoreCAM
import torch.nn.functional as F
from methods.gradcam import GradCAM, GradCAMpp
from methods.ham import HAM
from skimage import transform
import matplotlib.pyplot as plt

def find_image_path(caps_dir, participant_id, session_id, preprocessing):
    from os import path
    if preprocessing == "t1-linear":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    else:
        raise NotImplementedError

    return image_path

if __name__ == '__main__':
    # Load black box model for explanations
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    args = get_args()
    model = MAXNet(num_classes=2,
                    args = args)
    model = model.to(device)
    model.conv3.register_backward_hook(model.save_grad)
    model.conv4.register_backward_hook(model.save_grad)
    model.conv5.register_backward_hook(model.save_grad)
    model.load_state_dict(torch.load('model_state.pth', map_location=torch.device('cpu')))
    model = model.eval()

    insertion = CausalMetric(model, 'ins', 16900, substrate_fn=torch.zeros_like)
    deletion = CausalMetric(model, 'del', 16900, substrate_fn=torch.zeros_like)

    caps_dir = '/path/to/CAPS'
    image_path = find_image_path(caps_dir, 'sub-136S0300', 'ses-M01', 't1-linear')
    image_nii = nib.load(image_path)
    image = image_nii.get_data()
    # Function that opens image from disk, normalizes it and converts to tensor
    read_tensor = transforms.Compose([
        MinMaxNormalization(),
        transforms.ToTensor(),
        lambda x: torch.unsqueeze(x, 0)
    ])

    img = read_tensor(image)

    ####################### RISE ##########################
    # explainer = RISE(model, (169, 208, 179), N=25, p1=0.1)
    # explainer.generate_masks(N=5000, s=10, p1=0.1)
    # activation_map = explainer(img[0].to(device))[1]
    ####################### RISE ##########################

    ####################### score CAM ##########################
    # maxnet_model_dict = dict(type='maxnet', arch=model, layer_name='conv5', input_size=(169, 208, 179))
    # maxnet_scorecam = ScoreCAM(maxnet_model_dict)
    # activation_map = maxnet_scorecam(img.to(device), class_idx = 0)
    ####################### score CAM ##########################

    ####################### Grad CAM ##########################
    # extractor = GradCAM(model, 'conv5')
    # model.zero_grad()
    # scores, _, _, _ = model(img.to(device))
    # _, class_idx = torch.max(scores.data, 1)
    # activation_map = extractor(0, scores)
    # activation_map = torch.unsqueeze(activation_map, 0)
    # activation_map = F.interpolate(activation_map, size=(169, 208, 179), mode='trilinear', align_corners=False)
    ####################### Grad CAM ##########################

    ####################### Grad CAMpp ##########################
    # extractor = GradCAMpp(model, 'conv5')
    # model.zero_grad()
    # scores, _, _, _ = model(img.to(device))
    # _, class_idx = torch.max(scores.data, 1)
    # activation_map = extractor(0, scores)
    # activation_map = torch.unsqueeze(activation_map, 0)
    # activation_map = F.interpolate(activation_map, size=(169, 208, 179), mode='trilinear', align_corners=False)
    ####################### Grad CAMpp ##########################

    # ####################### HAM ##########################
    intermediate_layers = args.intermediate_layers.split(',')
    cam = HAM(model, intermediate_layers)
    activation_map = cam(img.to(device), device, intermediate_layers, 0)
    ####################### HAM ##########################


    h = deletion.single_run(img, activation_map, verbose=1, device=device, save_to='del')
    h = insertion.single_run(img, activation_map, verbose=1)