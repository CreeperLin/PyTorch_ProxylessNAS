#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import logging
import argparse

from utils.routine import augment
from utils.hparam import HParam
from utils.model import get_model
import utils
import genotypes as gt

from dataset.dataloader import load_data

def main():
    parser = argparse.ArgumentParser(description='Proxyless-NAS augment')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c','--config',type=str, default='./config/default.yaml',
                        help="yaml config file")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-d','--device',type=str,default="all",
                        help="override device ids")
    parser.add_argument('-g','--genotype',type=str,default=None,
                        help="override genotype file")
    args = parser.parse_args()

    hp = HParam(args.config)

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    out_dir = os.path.join(pt_path, args.name)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)

    logger = utils.get_logger(log_dir, args.name)

    if utils.check_config(hp, args.name):
        raise Exception("Config error.")

    writer = utils.get_writer(log_dir, hp.log.writer)
    
    dev, dev_list = utils.init_device(hp.device, args.device)

    trn_loader = load_data(hp.augment.data, validation=False)
    val_loader = load_data(hp.augment.data, validation=True)

    gt.set_primitives(hp.genotypes)

    # load genotype
    genotype = utils.get_genotype(hp.augment, args.genotype)
    
    model, arch = get_model(hp.model, dev, dev_list, genotype)

    augment(out_dir, args.chkpt, trn_loader, val_loader, model, writer, logger, dev, hp.augment)


if __name__ == '__main__':
    main()