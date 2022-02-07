import argparse
import torch
from model.model import MAXNet
import os
import pickle

def load_state(directory_path: str, device):
    with open(directory_path + '/model.pkl', 'rb') as f:
        model = pickle.load(f)
        state = torch.load(directory_path + '/model_state.pth', map_location=device)
        model.load_state_dict(state)
    return model

def init_model(model: MAXNet, optimizer, scheduler, device, args: argparse.Namespace):
    epoch = 1
    mean = 0.5
    std = 0.1

    torch.nn.init.normal_(model.feature_layer.feature_vectors, mean=mean, std=std)
    model._add_on.apply(init_weights_xavier)
    return model, epoch

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

def init_weights_kaiming(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')