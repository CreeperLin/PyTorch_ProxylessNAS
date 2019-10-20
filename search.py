#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import logging
import argparse

from utils.routine import search
from utils.hparam import HParam
from utils.model import get_model
import utils

from dataset.dataloader import load_data
import genotypes as gt

def main():
    parser = argparse.ArgumentParser(description='Proxyless-NAS arch search')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c','--config',type=str, default='./config/default.yaml',
                        help="yaml config file")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-d','--device',type=str,default="all",
                        help="override device ids")
    args = parser.parse_args()

    hp = HParam(args.config)
    if utils.check_config(hp, args.name):
        raise Exception("Config error.")
    hp_str = hp.to_string()

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    out_dir = os.path.join(pt_path, args.name)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)

    logger = utils.get_logger(log_dir, args.name)

    writer = utils.get_writer(log_dir, hp.log.writer)
    
    dev, dev_list = utils.init_device(hp.device, args.device)

    trn_loader, val_loader = load_data(hp.search.data, validation=False)

    gt.set_primitives(hp.genotypes)

    model, arch = get_model(hp.model, dev, dev_list)    

    search(out_dir, args.chkpt, trn_loader, val_loader, model, arch, writer, logger, dev, hp.search)


if __name__ == '__main__':
    main()