#-*- coding: utf-8 -*-
from __future__ import absolute_import
import time

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
from datetime import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from configs.get_config import load_config
from models import *
from datasets import *
from losses import *
from lib.core_function import validate, train, test, train_simple, validate_simple
from logs.logger import Logger, LOG_DIR
from lib.optimizers.sam import SAM
from lib.scheduler.linear_decay import LinearDecayLR
import json
import logging


def args_parser(args=None):
    parser = argparse.ArgumentParser("Training process...")
    parser.add_argument('--cfg', help='Config file', required=True)
    parser.add_argument('--alloc_mem', '-a',  help='Pre allocating GPU memory', action='store_true')
    return parser.parse_args(args)


if __name__=='__main__':
    if len(sys.argv[1:]):
        args = sys.argv[1:]
    else:
        args = None

    simple_models = {"EFN", "XCEP", "SPSL", "SBI"}
    
    args = args_parser(args)
    cfg = load_config(args.cfg)
    logger = Logger(task=f'training_{cfg.TASK}')
    start_time = datetime.now()
    with open(os.path.join(LOG_DIR, "_".join(str(start_time).split(" ")) + ".json"), "w") as f:
        json.dump(dict(cfg), f)

    #Seed
    seed = cfg.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Allocate memory
    if args.alloc_mem:
        mem_all_tensors = torch.rand(60,10000,10000)
        mem_all_tensors.to('cuda:0')

    #Configuing GPU devices
    devices = torch.device('cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if 'gpus' in cfg.TRAIN.gpus and cfg.TRAIN.gpus is not None:
        #Only support a single gpu for training now
        devices = torch.device('cuda:1')

    if cfg.MODEL.type == "EFN":
        from models.networks.efficientNet import get_model_params
        from models.networks.pose_efficientNet import EfficientNet
        block_args, global_params = get_model_params("efficientnet-b4", override_params={"num_classes": 1})
        model = EfficientNet(blocks_args=block_args, global_params=global_params, model_name="efficientnet-b4")
        # logger.info(f"loaded model {model}")
        start_epoch = 0
    elif cfg.MODEL.type == "XCEP":
        from models.networks.xception import Xception_simple
        model = Xception_simple(num_classes=1)
        logger.info(f"loaded model {model.name}")
        state_dict = torch.load(cfg.TEST.pretrained)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("last_linear", "fc"): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained weights loaded")
        start_epoch = 0
    elif cfg.MODEL.type == "SBI":
        from models.networks.sbi import Detector
        model = Detector()
        logger.info(f"loaded model SBI")
        model.net.load_state_dict(
            {k[4:]: v for k, v in torch.load(cfg.TRAIN.pretrained)["model"].items() if k.startswith("net.")}
        )
        # model = load_pretrained(model, cfg.TRAIN.pretrained)
        logger.info("Pretrained weights loaded")
        start_epoch = 0
    else:
        model = build_model(cfg.MODEL, MODELS).cuda().to(torch.float64)
    
    #Loading Dataloader
    start_loading = time.time()
    val_dataset = build_dataset(cfg.DATASET, 
                                DATASETS, 
                                default_args=dict(split='val', config=cfg.DATASET))
    if cfg.DATASET.PMM.DEGRADATIONS.val_degradations_p != 0:
        from lib.pdm import utils_blindsr as blindsr
        val_dataset = blindsr.PDM_DatasetWrapper(val_dataset, cfg.DATASET.PMM.DEGRADATIONS, split="val")
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                shuffle=False,
                                pin_memory=cfg.DATASET.PIN_MEMORY,
                                num_workers=cfg.DATASET.NUM_WORKERS,
                                worker_init_fn=val_dataset.train_worker_init_fn,
                                collate_fn=val_dataset.train_collate_fn)
                                
    logger.info('Loading val dataloader successfully! -- {}'.format(time.time() - start_loading))
    logger.info(f"Total GPU memory: {(torch.cuda.get_device_properties(0).total_memory / 2**30):2.2f}GB")
    start_loading = time.time()
    train_dataset = build_dataset(cfg.DATASET,
                                  DATASETS,
                                  default_args=dict(split='train', config=cfg.DATASET))
    if cfg.DATASET.PMM.DEGRADATIONS.degradations_p != 0:
        from lib.pdm import utils_blindsr as blindsr
        train_dataset = blindsr.PDM_DatasetWrapper(train_dataset, cfg.DATASET.PMM.DEGRADATIONS)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                  shuffle=True,
                                  pin_memory=cfg.DATASET.PIN_MEMORY,
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  worker_init_fn=train_dataset.train_worker_init_fn,
                                  collate_fn=train_dataset.train_collate_fn)
    logger.info('Loading Train dataloader successfully! -- {}'.format(time.time() - start_loading))
    
    # start_loading = time.time()
    # test_dataset = build_dataset(cfg.DATASET, 
    #                              DATASETS,
    #                              default_args=dict(split='test', config=cfg.DATASET))
    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
    #                              shuffle=True,
    #                              pin_memory=cfg.DATASET.PIN_MEMORY,
    #                              num_workers=cfg.DATASET.NUM_WORKERS)
    # logger.info('Loading Test dataloader successfully! -- {}'.format(time.time() - start_loading))
    
    #Defining Loss function and Optimizer
    if cfg.TRAIN.loss.type == "BCE":
        critetion = torch.nn.BCEWithLogitsLoss()
    elif cfg.TRAIN.loss.type == "CE":
        critetion = torch.nn.CrossEntropyLoss()
    else:
        critetion = build_losses(cfg.TRAIN.loss, LOSSES, default_args=dict(cfg=cfg.TRAIN.loss)).cuda().to(torch.float64)

    if cfg.TRAIN.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=2e-5)
    elif cfg.TRAIN.optimizer == 'SAM':
        optimizer = SAM(model.parameters(), optim.Adam, lr=cfg.TRAIN.lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=1e-5, momentum=0.9)
        
    #Loading model
    if not cfg.MODEL.type in simple_models:
        model, optimizer, start_epoch = preset_model(cfg, model, optimizer=optimizer)

    if len(cfg.TRAIN.gpus) > 0:
        model = nn.DataParallel(model, device_ids=cfg.TRAIN.gpus).cuda().to(torch.float64)
    else:
        model = model.cuda().to(torch.float64)
    
    #Learning rate Scheduler
    if cfg.TRAIN.lr_scheduler.type == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **cfg.TRAIN.lr_scheduler)
    elif cfg.TRAIN.lr_scheduler.type == 'ReduceLROnPlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
    else:
        lr_scheduler = LinearDecayLR(optimizer, cfg.TRAIN.epochs, cfg.TRAIN.epochs//4, last_epoch=cfg.TRAIN.begin_epoch, booster=4)


    logging.info((lr_scheduler, optimizer.param_groups[0]["lr"]))
    #Enabling tensorboard
    writer = SummaryWriter('.tensorboard/{}_{}'.format(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'), cfg.TASK))
    
    trainIters = 0
    valIters = 0
    min_val_loss = 1e10
    max_val_acc = 0
    max_test_auc = 0
    metrics_base = cfg.METRICS_BASE # Combine heatmap + cls prediction to calculate accuracy

    if cfg.MODEL.type in simple_models:
        train_fn = train_simple
        val_fn = validate_simple
    else:
        train_fn = train
        val_fn = validate
    
    #Starting training process
    logger.info('Starting training process...')
    for epoch in range(start_epoch, cfg.TRAIN.epochs):
        logging.info((lr_scheduler, optimizer.param_groups[0]["lr"]))
        #Unfreezin backbone to update weights
        if cfg.TRAIN.freeze_backbone and epoch == cfg.TRAIN.warm_up:
            unfreeze_backbone(model)
        
        np.random.seed(seed + epoch)
        if epoch > 0 and cfg.DATA_RELOAD:
            logger.info(f'Reloading data for epoch {epoch}...')
            train_dataset._reload_data()
            train_dataloader = DataLoader(train_dataset, 
                                          batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                          shuffle=True,
                                          pin_memory=cfg.DATASET.PIN_MEMORY,
                                          num_workers=cfg.DATASET.NUM_WORKERS,
                                          worker_init_fn=train_dataset.train_worker_init_fn,
                                          collate_fn=train_dataset.train_collate_fn)

   
        loss_avg, acc_avg, trainIters = train_fn(cfg, 
                                            model, 
                                            critetion, 
                                            optimizer, 
                                            epoch, 
                                            train_dataloader, 
                                            logger, 
                                            writer, 
                                            devices, 
                                            trainIters, 
                                            metrics_base=metrics_base)
        if epoch % cfg.TRAIN.every_val_epochs == 0:
            loss_val, acc_val, valIters = val_fn(cfg, 
                                                   model, 
                                                   critetion, 
                                                   epoch, 
                                                   val_dataloader, 
                                                   logger, 
                                                   writer, 
                                                   devices, 
                                                   valIters, 
                                                   metrics_base=metrics_base)

            if ((acc_val.avg > max_val_acc)): # and (loss_val.avg < min_val_loss)):
                # Saving checkpoint 
                ckp_path = os.path.join(LOG_DIR, '{}_{}_model_best_{}.pth'.format(cfg.MODEL.type, cfg.TASK, \
                                                                                  #"_".join([f"{ko}:{ki}:{vi}" for ko,vo in dict(cfg.DATASET.PMM).items() for ki, vi in vo.items()]),
                                                                                  "_".join(str(start_time).split(" "))
                                                                                  ))
                save_model(path=ckp_path, epoch=epoch, model=model, optimizer=optimizer)
                min_val_loss = loss_val.avg
                max_val_acc = acc_val.avg
                logger.info(f'Saved best model at epoch --- {epoch}')

            ckp_path = os.path.join(LOG_DIR, '{}_{}_checkpoint_{}.pth'.format(cfg.MODEL.type, cfg.TASK, \
                                                                                  #"_".join([f"{ko}:{ki}:{vi}" for ko,vo in dict(cfg.DATASET.PMM).items() for ki, vi in vo.items()]),
                                                                                  "_".join(str(start_time).split(" "))
                                                                                  ))
            save_model(path=ckp_path, epoch=epoch, model=model, optimizer=optimizer)
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(loss_val.avg)
        else:
            lr_scheduler.step()
        
        if cfg.TRAIN.tensorboard:
            writer.add_scalar("hyperparameter/lr", optimizer.param_groups[0]["lr"], epoch)
