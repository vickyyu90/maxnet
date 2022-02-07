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
                disable_derivative_free_leaf_optim: bool,
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
    # Iterate through the data set to update leaves, prototypes and network
    for i, (data) in train_iter:
        xs, ys = data['image'].to(device), data['label'].to(device)
        # Make sure the model is in train mode
        model.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # predict
        pred, distances = model.forward(xs)
        cross_entropy = torch.nn.functional.cross_entropy(pred, ys)

        max_dist = (distances.shape[1])
        prototypes_of_correct_class = torch.t(pred[:, data['label']])  # .cuda()
        inverted_distances, _ = torch.max(torch.matmul(prototypes_of_correct_class, (max_dist - distances)), dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances) / distances.shape[1]

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max(torch.matmul(prototypes_of_wrong_class, (max_dist - distances)), dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes) / distances.shape[1]

        # calculate avg cluster cost
        avg_separation_cost = \
            torch.sum(torch.matmul(prototypes_of_wrong_class, distances), dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                dim=1)
        avg_separation_cost = torch.mean(avg_separation_cost)

        # evaluation statistics
        _, predicted = torch.max(pred.data, 1)
        n_examples += data['label'].size(0)
        n_correct += (predicted == data['label']).sum().item()

        n_batches += 1
        total_cross_entropy += cross_entropy.item()
        total_cluster_cost += cluster_cost.item()
        total_separation_cost += separation_cost.item()
        total_avg_separation_cost += avg_separation_cost.item()

        l1 = model.classifier.weight.norm(p=1)
        # compute gradient and do SGD step
        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
        
        # Compute the gradient
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
    print('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    print('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    print('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    print('\tl1: \t\t{0}'.format(model.classifier.weight.norm(p=1).item()))

    train_info['cluster loss'] = total_cluster_cost / n_batches
    train_info['separation loss'] = total_separation_cost / n_batches
    train_info['train_accuracy'] = n_correct / n_examples
    return train_info