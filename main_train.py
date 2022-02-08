from model.model import MAXNet

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_model
from util.save import *
from model.train import train_epoch
from model.test import eval

import torch
from shutil import copy
from copy import deepcopy

def run_train(args=None):
    args = args or get_args()
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    log.create_log('log_epoch_data', 'epoch', 'test_acc', 'mean_train_acc')
    # save arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        
    # Log device
    log.log_message('Device: '+str(device))

    # create log entries
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # get dataloaders
    trainloader, valoader, testloader = get_dataloaders(args)

    model = MAXNet(num_classes=2,
                    args = args)
    model = model.to(device=device)
    model.load_state_dict(torch.load('model_state.pth', map_location=torch.device('cpu')))

    # get optimizer
    optimizer, params_to_freeze, params_to_train = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    model, epoch = init_model(model, optimizer, scheduler, device, args)
    
    model.save(f'{log.checkpoint_dir}/model_init')

    best_train_acc = 0.
    best_test_acc = 0.

    if epoch < args.epochs + 1:
        # Train
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s"%str(epoch))
            log.log_learning_rates(optimizer)
            
            # Train
            train_info = train_epoch(model, trainloader, optimizer, epoch, device, log, log_prefix)
            save_model(model, optimizer, scheduler, epoch, log, args)

            # Evaluate
            if args.epochs > 100:
                if epoch % 10 == 0 or epoch == args.epochs:
                    eval_info = eval(model, testloader, epoch, device, log)
                    test_acc = eval_info['test_accuracy']
                    best_test_acc = save_best_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                    log.log_values('log_epoch_data', epoch, eval_info['test_accuracy'], train_info['train_accuracy'])
                else:
                    log.log_values('log_epoch_data', epoch, "n.a.", train_info['train_accuracy'])
            else:
                eval_info = eval(model, testloader, epoch, device, log)
                test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'])
            
            scheduler.step()
 
    else:
        # evaluate
        eval_info = eval(model, testloader, epoch, device, log)
        test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.")

    log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n"%(str(best_train_acc), str(best_test_acc)))
    trained_model = deepcopy(model)

    
    return trained_model.to('cpu'), test_acc


if __name__ == '__main__':

    args = get_args()
    run_train(args)