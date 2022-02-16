from tqdm import tqdm
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from model.model import MAXNet

from util.log import Log

def train_epoch(model: MAXNet,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:
    
    model = model.to(device)
    # temp variables
    train_info = dict()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    # create a progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+' %s'%epoch,
                    ncols=0)
    model.train()

    for i, (data) in train_iter:
        xs, ys = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # predict
        pred, distances = model.forward(xs)
        cross_entropy = torch.nn.functional.cross_entropy(pred, ys)

        max_dist = (distances.shape[1])
        features_target = torch.t(pred[:, data['label']])  # .cuda()
        reversed_distances, _ = torch.max(torch.matmul(features_target, (max_dist - distances)), dim=1)
        cluster_cost = torch.mean(max_dist - reversed_distances) / distances.shape[1]

        # calculate separation cost
        features_nontarget = 1 - features_target
        reversed_distances_nontarget, _ = \
            torch.max(torch.matmul(features_nontarget, (max_dist - distances)), dim=1)
        separation_cost = torch.mean(max_dist - reversed_distances_nontarget) / distances.shape[1]

        avg_sep_cost = \
            torch.sum(torch.matmul(features_nontarget, distances), dim=1) / torch.sum(features_nontarget,
                                                                                dim=1)
        avg_sep_cost = torch.mean(avg_sep_cost)

        _, predicted = torch.max(pred.data, 1)
        n_examples += data['label'].size(0)
        n_correct += (predicted == data['label']).sum().item()

        n_batches += 1
        total_cross_entropy += cross_entropy.item()
        total_cluster_cost += cluster_cost.item()
        total_separation_cost += separation_cost.item()
        total_avg_separation_cost += avg_sep_cost.item()

        l1 = model.classifier.weight.norm(p=1)
        loss = cross_entropy + 0.6 * cluster_cost - 0.06 * separation_cost + 1e-4 * l1

        loss.backward()
        # Update model parameters
        optimizer.step()

        del data
        del xs
        del ys
        del predicted
        del distances
        del pred

    print('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    print('\tcluster cost: \t{0}'.format(total_cluster_cost / n_batches))
    print('\tseparation cost:\t{0}'.format(total_separation_cost / n_batches))
    print('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    print('\tl1: \t\t{0}'.format(model.classifier.weight.norm(p=1).item()))

    train_info['cluster loss'] = total_cluster_cost / n_batches
    train_info['separation loss'] = total_separation_cost / n_batches
    train_info['train_accuracy'] = n_correct / n_examples
    return train_info