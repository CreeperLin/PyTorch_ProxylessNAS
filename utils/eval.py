# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import itertools

# from utils.hparam import load_hparam_str
import utils
import genotypes as gt
from visualize import plot
from profile.profiler import tprof
from models.nas_modules import NASModule

def augment(out_dir, chkpt_path, train_data, valid_data, model, writer, logger, device, config):
    
    valid_loader = DataLoader(valid_data,
        batch_size=config.val_batch_size,
        num_workers=config.workers,
        shuffle=False, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_data,
        batch_size=config.trn_batch_size,
        num_workers=config.workers,
        shuffle=True, pin_memory=True, drop_last=True)
    
    w_optim = utils.get_optim(model.weights(), config.w_optim)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_optim.lr_min)

    init_epoch = -1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        w_optim.load_state_dict(checkpoint['w_optim'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']

    else:
        logger.info("Starting new training run")

    # training loop
    logger.info('training begin')
    best_top1 = 0.
    tot_epochs = config.epochs
    for epoch in itertools.count(init_epoch+1):
        if epoch == tot_epochs: break

        drop_prob = config.drop_path_prob * epoch / tot_epochs
        model.drop_path_prob(drop_prob)

        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        # training
        train(train_loader, model, writer, logger, w_optim, epoch, tot_epochs, lr, device, config)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, device, cur_step, config) 

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        
        if config.save_freq == 0 or epoch % config.save_freq != 0:
            print("")
            continue
        
        try:
            save_path = os.path.join(out_dir, 'chkpt_%03d.pt' % (epoch+1))
            torch.save({
                'model': model.state_dict(),
                'w_optim': w_optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                # 'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
        except Exception as e:
            logger.error("Save checkpoint failed: "+str(e))

        print("")
        
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    tprof.stat_acc('model_'+NASModule.get_device()[0])

def train(train_loader, model, writer, logger, optim, epoch, tot_epochs, lr, device, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    logger.info("Epoch {} LR {}".format(epoch, lr))
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    eta_m = utils.ETAMeter(tot_epochs, epoch, len(train_loader))
    eta_m.start()
    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optim.zero_grad()
        tprof.timer_start('augment-train')
        logits = model(X)
        tprof.timer_stop('augment-train')
        loss = model.criterion(logits, y)
        # logits, aux_logits = model(X)
        # if config.aux_weight > 0.:
            # loss += config.aux_weight * model.criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optim.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            eta = eta_m.step(step)
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) | ETA: {eta}".format(
                    epoch+1, tot_epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5, eta=utils.format_time(eta)))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.print_stat('augment-train')

def validate(valid_loader, model, writer, logger, epoch, tot_epochs, device, cur_step, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            tprof.timer_start('validate')
            logits = model(X)
            tprof.timer_stop('validate')
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, tot_epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.print_stat('validate')

    return top1.avg
