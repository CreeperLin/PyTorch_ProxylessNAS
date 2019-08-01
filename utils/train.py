# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import adabound
import itertools
import traceback

# from utils.hparam import load_hparam_str
import utils
from utils.eval import validate
from architect import DARTSArchitect, BinaryGateArchitect
from visualize import plot

def search(out_dir, chkpt_path, train_data, valid_data, model, writer, logger, device, config):

    if config.split:
        data = train_data
        n_train = len(data)
        split = int(n_train * config.w_data_ratio)
        print('# of w/a data: {}/{}'.format(split, n_train-split))
        indices = list(range(n_train))
        w_train_sampler = SubsetRandomSampler(indices[:split])
        a_train_sampler = SubsetRandomSampler(indices[split:])
        w_train_loader = DataLoader(data,
                        batch_size=config.trn_batch_size,
                        sampler=w_train_sampler,
                        num_workers=config.workers,
                        pin_memory=True)
        a_train_loader = DataLoader(data,
                        batch_size=config.trn_batch_size,
                        sampler=a_train_sampler,
                        num_workers=config.workers,
                        pin_memory=True)
        if valid_data is None:
            valid_loader = a_train_loader
        else:
            valid_loader = DataLoader(valid_data,
                batch_size=config.val_batch_size,
                num_workers=config.num_workers,
                shuffle=False, pin_memory=True, drop_last=False)
    else:
        a_train_loader = DataLoader(valid_data,
            batch_size=config.trn_batch_size,
            num_workers=config.num_workers,
            shuffle=False, pin_memory=True, drop_last=False)
        w_train_loader = DataLoader(train_data,
            batch_size=config.trn_batch_size,
            num_workers=config.num_workers,
            shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = a_train_loader
    
    w_optim = utils.get_optim(model.weights(), config.w_optim)
    a_optim = utils.get_optim(model.alphas(), config.a_optim)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_optim.lr_min)

    init_epoch = -1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        w_optim.load_state_dict(checkpoint['w_optim'])
        a_optim.load_state_dict(checkpoint['a_optim'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']

        # if hp_str != checkpoint['hp_str']:
        #     logger.warning("New hparams are different from checkpoint.")
        #     logger.warning("Will use new hparams.")
        # hp = load_hparam_str(hp_str)
    else:
        logger.info("Starting new training run")
    
    # architect = DARTSArchitect(model, config.w_optim.momentum, config.w_optim.weight_decay)
    architect = BinaryGateArchitect(model, config.w_optim.momentum, config.w_optim.weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in itertools.count(init_epoch+1):
        if epoch == config.epochs: break

        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(w_train_loader, a_train_loader, model, writer, logger, architect, w_optim, a_optim, lr, epoch, device, config)

        # validation
        cur_step = (epoch+1) * len(w_train_loader)
        top1 = validate(valid_loader, model, writer, logger, epoch, device, cur_step, config) 

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        # caption = "Epoch {}".format(epoch+1)
        # plot(genotype.normal, plot_path + "-normal", caption)
        # plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False

        save_path = os.path.join(out_dir, 'chkpt_%03d.pt' % epoch)
        torch.save({
            'model': model.state_dict(),
            'w_optim': w_optim.state_dict(),
            'a_optim': a_optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            # 'hp_str': hp_str,
        }, save_path)
        logger.info("Saved checkpoint to: %s" % save_path)

        print("")
        
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, writer, logger, architect, w_optim, a_optim, lr, epoch, device, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    tr_ratio = len(train_loader) // len(valid_loader)
    val_iter = iter(valid_loader)
    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        if step % tr_ratio == 0:
            # phase 2. architect step (alpha)
            val_X, val_y = next(val_iter)   
            val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
            architect.step(trn_X, trn_y, val_X, val_y, lr, w_optim, a_optim)

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
