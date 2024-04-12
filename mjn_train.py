import torch
import time
import logging

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

# mjn
import re

# mjn
def get_layer(param_name):
    match = re.search(r"\.layers\.\d+", param_name)
    if not match:
        return False
    layer = match.group()
    return layer

# mjn - the following function helps implement the Auto-RGN algorithm
def calculate_rgn(model):
    param_list = []
    rgn_values = {}
    for name, param in model.named_parameters():
        layer = get_layer(name)
        param_list.append(layer)
        if layer != False:
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                param_norm = param.data.norm(2)
                rgn = grad_norm / (param_norm + 1e-7)  # Avoid division by zero
                if layer in rgn_values:
                    layer[rgn_values.append(rgn.item())] += rgn
                else:
                    layer[rgn_values.append(rgn.item())] = rgn
    return param_list, rgn_values

# mjn - the following function helps implement the Auto-RGN algorithm
def adjust_learning_rates(optimizer, param_list, rgn_values):
    # Normalize RGN values
    max_rgn = max(rgn_values.values())
    min_rgn = min(rgn_values.values())
    normalized_rgns = {}
    for layer in rgn_values:
        normalized_rgns[layer] = (rgn_values[layer] - min_rgn) / (max_rgn - min_rgn + 1e-7)

    print(normalized_rgns)

    # Adjust learning rates for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        if param_list[i] != False:
            initial_lr = param_group['initial_lr']
            param_group['lr'] = initial_lr * normalized_rgns[param_list[i]]

def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()

        # mjn - # Calculate RGN and adjust learning rates
        param_list, rgn_values = calculate_rgn(model)
        adjust_learning_rates(optimizer, param_list, rgn_values)

        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))