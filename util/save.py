import torch
import os
import argparse
from model.model import MAXNet
from util.log import Log

def save_model(model: MAXNet, optimizer, scheduler, epoch: int, log: Log, args: argparse.Namespace):
    model.eval()
    # Save latest model
    model.save(f'{log.checkpoint_dir}/latest')
    model.save_state(f'{log.checkpoint_dir}/latest')
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/latest/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/latest/scheduler_state.pth')

    # Save model every 10 epochs
    if epoch == args.epochs or epoch % 10 == 0:
        model.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        model.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/scheduler_state.pth')

def save_best_model(model: MAXNet, optimizer, scheduler, best_test_acc: float, test_acc: float, log: Log):
    model.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        model.save(f'{log.checkpoint_dir}/best_test_model')
        model.save_state(f'{log.checkpoint_dir}/best_test_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_test_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_test_model/scheduler_state.pth')
    return best_test_acc

def save_model_details(model: MAXNet, optimizer, scheduler, description: str, log: Log):
    model.eval()
    # Save model with description
    model.save(f'{log.checkpoint_dir}/'+description)
    model.save_state(f'{log.checkpoint_dir}/'+description)
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/'+description+'/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/'+description+'/scheduler_state.pth')
