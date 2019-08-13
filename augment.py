#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import logging
import argparse

from utils.train import augment
from utils.hparams import HParam
from utils.writer import MyWriter
from utils import check_config
from utils import get_logger
from utils import get_model

from dataset.dataloader import load_data

def main():
    parser = argparse.ArgumentParser(description='augment Proxyless-NAS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c','--config',type=str, default='./config/default.yaml',
                        help="yaml config file")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-g','--gpus',type=str,default="all",
                        help="override gpu ids")
    args = parser.parse_args()

    hp = HParam(args.config)

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    out_dir = os.path.join(pt_path, args.name)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(log_dir, args.name)

    if check_config(hp, logger):
        raise Exception("Config error.")

    writer = get_writer(log_dir)
    writer.add_text('config', hp_str, 0)
    
    dev, dev_list = init_device(hp.device, args.gpus)

    train_data = load_data(hp.data.augment)
    val_data = load_data(hp.data.augment, True)

    gt.PRIMITIVES_DEFAULT = hp.genotypes
    print(gt.PRIMITIVES_DEFAULT)

    # load genotype
    g_str = config.genotype
    if g_str == '':
        genotype = gt.from_file(config.gt_file)
    else:
        genotype = gt.from_str(g_str)
    
    model, arch = get_model(hp.model, dev, dev_list, genotype)

    augment(out_dir, args.chkpt, train_data, val_data, model, writer, logger, dev, hp.valid)


if __name__ == '__main__':
    main()