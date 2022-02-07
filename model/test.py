import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from model.model import MAXNet
from util.log import Log


@torch.no_grad()
def eval(model: MAXNet,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        sampling_strategy: str = 'distributed',
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    model = model.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((model._num_classes, model._num_classes), dtype=int)
    model.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)

    # Iterate through the test set
    for i, (data) in test_iter:
        xs, ys = data['image'].to(device), data['label'].to(device)
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out, test_info = model.forward(xs)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((model._num_classes, model._num_classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        acc = acc_from_cm(cm_batch)
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc:.3f}'
        )

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        if sampling_strategy != 'distributed':
            info['out_leaf_ix'] += test_info['out_leaf_ix']
        del out
        del ys_pred
        del test_info

    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    log.log_message("\nEpoch %s - Test accuracy with %s: "%(epoch, sampling_strategy)+str(info['test_accuracy']))
    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
