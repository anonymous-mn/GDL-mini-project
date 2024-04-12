import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

# mjn - for loading pretrained model
from torch_geometric.graphgym.checkpoint import load_ckpt, get_ckpt_path

# mjn - to train with our custom method, i.e., with partial freezing
from mjn_train import train as mjn_train

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':

    # mjn - specify layers to freeze:
    layers_to_freeze = [] # e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    use_AUTO_RGN = False

    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()


        # mnj - doing it for several layer freezings
        #for freezer in range(4):
        # mjn - that is for now
        
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )

        # mjn - load model
        #saved_path = os.path.abspath('results/zinc/model.pt')
        #model.load_state_dict(torch.load(saved_path))
        
        optimizer = create_optimizer(model.parameters(),
                                    new_optimizer_config(cfg))

        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        # mjn - in the case of fine-tuning (i.e., if the loaded cfg file is for fine-tuning), load the respective model:
        if "-FT" in cfg.dataset.format:
            if cfg.dataset.name == "CIFAR10"
                path = os.path.abspath('results/cifar10-GPS+RWSE/0/ckpt/19.ckpt')
            elif cfg.dataset.name == "CLUSTER"
                path = os.path.abspath('results/cluster-GPS-ESLapPE/0/ckpt/19.ckpt')      
            print(path)
            MODEL_STATE = 'model_state'
            OPTIMIZER_STATE = 'optimizer_state'
            SCHEDULER_STATE = 'scheduler_state'
            ckpt = torch.load(path)
            model.load_state_dict(ckpt[MODEL_STATE])
            if optimizer is not None and OPTIMIZER_STATE in ckpt:
                optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
            if scheduler is not None and SCHEDULER_STATE in ckpt:
                scheduler.load_state_dict(ckpt[SCHEDULER_STATE])
            
            # mjn - freeze prescribed layers
            freeze_layers = ['model.layers.' + str(layer) + '.' for layer in layers_to_freeze]
            #if freezer == 0:
            #freeze_layers = ['model.layers.0.', 'model.layers.1.', 'model.layers.2.', 'model.layers.3.', 'model.layers.4.', 'model.layers.5.', 'model.layers.6.', 'model.layers.7.', 'model.layers.8.', 'model.layers.9.']#, 'model.layers.10.', 'model.layers.11.', 'model.layers.12.']
            #if freezer == 1:
            #freeze_layers = ['model.layers.3.', 'model.layers.4.', 'model.layers.5.', 'model.layers.6.', 'model.layers.7.', 'model.layers.8.', 'model.layers.9.', 'model.layers.10.', 'model.layers.11.', 'model.layers.12.', 'model.layers.13.', 'model.layers.14.', 'model.layers.15.']
            #if freezer == 2:
            #    freeze_layers = ['model.layers.7', 'model.layers.8', 'model.layers.9']
            print(freeze_layers)
            for name, param in model.named_parameters():
                for layer_name in freeze_layers:
                    #if not (('model.layers.4' in name) or ('model.layers.5' in name)):
                    if layer_name in name:
                        param.requires_grad = False
                        print(name + ' is freezed during fine-tuning')
            # mjn - that is for now

        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            
            # mjn - activate AUTO-RGN during fine-tuning:
            if use_AUTO_RGN:
                mjn_train(model, datamodule, logger=True)
            else:
                train(model, datamodule, logger=True)
            
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                    scheduler)

        # mjn - save model
        #save_path = os.path.abspath('pretrained/pattern/model.pt')
        #torch.save(model.state_dict(), save_path)

    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
