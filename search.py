#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import logging
import argparse

from utils.train import search
from utils.hparam import HParam
from utils.model import get_model
from utils import check_config, get_logger, get_writer, init_device, param_size

from dataset.dataloader import load_data
import genotypes as gt

def main():
    parser = argparse.ArgumentParser(description='train Proxyless-NAS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c','--config',type=str, default='./config/default.yaml',
                        help="yaml config file")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-g','--gpus', type=str, default=None,
                        help="override gpu ids")
    args = parser.parse_args()

    hp = HParam(args.config)
    if check_config(hp, args.name):
        raise Exception("Config error.")
    hp_str = hp.to_string()

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    out_dir = os.path.join(pt_path, args.name)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(log_dir, args.name)

    writer = get_writer(log_dir)
    writer.add_text('config', hp_str, 0)
    
    # train_dl = load_data(hp.data, val=False)
    # val_dl = load_data(hp.data, val=True)
    dev, dev_list = init_device(hp.device, args.gpus)

    train_data = load_data(hp.data.search)

    gt.PRIMITIVES_DEFAULT = hp.genotypes
    print(gt.PRIMITIVES_DEFAULT)

    model = get_model(hp.model, dev, dev_list)
    mb_params = param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    search(out_dir, args.chkpt, train_data, None, model, writer, logger, dev, hp.train)


if __name__ == '__main__':
    main()