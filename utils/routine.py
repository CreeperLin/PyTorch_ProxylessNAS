# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import utils
from visualize import plot
from utils.profiling import tprof
import genotypes as gt
from models.nas_modules import NASModule

def save_checkpoint(out_dir, model, w_optim, a_optim, lr_scheduler, epoch, logger):
    try:
        save_path = os.path.join(out_dir, 'chkpt_%03d.pt' % (epoch+1))
        torch.save({
            'model': model.state_dict(),
            'arch': NASModule.nasmod_state_dict(),
            'w_optim': w_optim.state_dict(),
            'a_optim': a_optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, save_path)
        logger.info("Saved checkpoint to: %s" % save_path)
    except Exception as e:
        logger.error("Save checkpoint failed: "+str(e))

def save_genotype(out_dir, genotype, epoch, logger):
    try:
        logger.info("genotype = {}".format(genotype))
        save_path = os.path.join(out_dir, 'gene_{:03d}.gt'.format(epoch+1))
        gt.to_file(genotype, save_path)
        logger.info("Saved genotype to: %s" % save_path)
    except:
        logger.error("Save genotype failed")


def search(out_dir, chkpt_path, w_train_loader, a_train_loader, model, arch, writer, logger, device, config):
    valid_loader = a_train_loader
    
    w_optim = utils.get_optim(model.weights(), config.w_optim)
    a_optim = utils.get_optim(model.alphas(), config.a_optim)

    warmup_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.warmup_epochs, eta_min=config.w_optim.lr_min)

    init_epoch = -1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        NASModule.nasmod_load_state_dict(checkpoint['arch'])
        w_optim.load_state_dict(checkpoint['w_optim'])
        a_optim.load_state_dict(checkpoint['a_optim'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']
    else:
        logger.info("Starting new training run")
    
    architect = arch(config, model)

    # warmup training loop
    logger.info('begin warmup training')
    try:
        tot_epochs = config.warmup_epochs
        for epoch in itertools.count(init_epoch+1):
            if epoch == tot_epochs: break

            warmup_lr_scheduler.step()
            lr = warmup_lr_scheduler.get_lr()[0]

            # training
            train(w_train_loader, None, model, writer, logger, architect, w_optim, a_optim, lr, epoch, tot_epochs, device, config)

            # validation
            cur_step = (epoch+1) * len(w_train_loader)
            top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, device, cur_step, config)

            print("")
    except KeyboardInterrupt:
        print('skipped')
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_optim.lr_min, last_epoch=0)
    
    save_checkpoint(out_dir, model, w_optim, a_optim, lr_scheduler, init_epoch, logger)
    save_genotype(out_dir, model.genotype(), init_epoch, logger)

    # training loop
    logger.info('begin w/a training')
    best_top1 = 0.
    tot_epochs = config.epochs
    for epoch in itertools.count(init_epoch+1):
        if epoch == tot_epochs: break

        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(w_train_loader, a_train_loader, model, writer, logger, architect, w_optim, a_optim, lr, epoch, tot_epochs, device, config)

        # validation
        cur_step = (epoch+1) * len(w_train_loader)
        top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config) 

        # genotype
        genotype = model.genotype()
        save_genotype(out_dir, genotype, epoch, logger)
        
        # genotype as a image
        if config.plot:
            for i, dag in enumerate(model.dags()):
                plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
                caption = "Epoch {} - DAG {}".format(epoch+1, i)
                plot(genotype.dag[i], dag, plot_path + "-dag_{}".format(i), caption)
        
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype

        if config.save_freq != 0 and epoch % config.save_freq == 0:
            save_checkpoint(out_dir, model, w_optim, a_optim, lr_scheduler, epoch, logger)

        print("")
        
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    tprof.stat_acc('model_'+NASModule.get_device()[0])
    gt.to_file(best_genotype, os.path.join(out_dir, 'best.gt'))


def augment(out_dir, chkpt_path, train_loader, valid_loader, model, writer, logger, device, config):
    
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

    logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_size(model), utils.param_count(model)))
    
    # training loop
    logger.info('begin training')
    best_top1 = 0.
    tot_epochs = config.epochs
    for epoch in itertools.count(init_epoch+1):
        if epoch == tot_epochs: break

        drop_prob = config.drop_path_prob * epoch / tot_epochs
        model.drop_path_prob(drop_prob)

        lr = lr_scheduler.get_lr()[0]

        # training
        train(train_loader, None, model, writer, logger, None, w_optim, None, lr, epoch, tot_epochs, device, config)
        lr_scheduler.step()

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config) 

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        
        if config.save_freq != 0 and epoch % config.save_freq == 0:
            save_checkpoint(out_dir, model, w_optim, None, lr_scheduler, epoch, logger)

        print("")
        
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    tprof.stat_acc('model_'+NASModule.get_device()[0])


def train(train_loader, valid_loader, model, writer, logger, architect, w_optim, a_optim, lr, epoch, tot_epochs, device, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    if not valid_loader is None:
        tr_ratio = len(train_loader) // len(valid_loader)
        val_iter = iter(valid_loader)

    eta_m = utils.ETAMeter(tot_epochs, epoch, len(train_loader))
    eta_m.start()
    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 1. child network step (w)
        w_optim.zero_grad()
        tprof.timer_start('train')
        loss, logits = model.loss(trn_X, trn_y, config.aux_weight)
        tprof.timer_stop('train')
        loss.backward()
        # gradient clipping
        if config.w_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        
        # phase 2. architect step (alpha)
        if not valid_loader is None and step % tr_ratio == 0:
            try:
                val_X, val_y = next(val_iter)
            except:
                val_iter = iter(valid_loader)
                val_X, val_y = next(val_iter)
            val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
            tprof.timer_start('arch')
            architect.step(trn_X, trn_y, val_X, val_y, lr, w_optim, a_optim)
            tprof.timer_stop('arch')

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step !=0 and step % config.print_freq == 0 or step == len(train_loader)-1:
            eta = eta_m.step(step)
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) | ETA: {eta}".format(
                    epoch+1, tot_epochs, step, len(train_loader)-1, lr, losses=losses,
                    top1=top1, top5=top5, eta=utils.format_time(eta)))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.stat_acc('model_'+NASModule.get_device()[0])
    tprof.print_stat('train')
    tprof.print_stat('arch')


def validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config):
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

            if step !=0 and step % config.print_freq == 0 or step == len(valid_loader)-1:
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
