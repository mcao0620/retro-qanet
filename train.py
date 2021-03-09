"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    sketchy_device, args.sketchy_gpu_ids = util.get_available_devices() 
    args.sketchy_gpu_ids = args.sketchy_gpu_ids([0:len(args.sketchy_gpu_ids)])  #We use half of the available devices given we have two models to train
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids_sketchy)) 

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    # Get Sketchy model
    log.info('Building model...')
    sketchy_model = #QANET SKETCHY MODEL INTITIATION
    sketchy_model = nn.DataParallel(sketchy_model, args.sketchy_gpu_ids)
    if args.load_path_s:
        log.info(f'Loading checkpoint from {args.load_path_s}...')
        sketchy_model, sketchy_step = util.load_model(sketchy_model, args.load_path_s, args.sketchy_gpu_ids)
    else:
        sketchy_step = 0
    sketchy_model = sketchy_model.to(sketchy_device)
    sketchy_model.train()
    sketchy_ema = util.EMA(sketchy_model, args.ema_decay_s)
     
    #check again for valid devices
    intensive_device, args.intensive_gpu_ids = util.get_available_devices()
    args.intensive_gpu_ids = args.intensive_gpu_ids[:,len(args.intensive_gpu_ids)/2]

    # Get Intensive model
    log.info('Building model...')
    intensive_model = #QANET INTENSIVE MODEL INTITIATION
    intensive_model = nn.DataParallel(intensive_model, args.intensive_gpu_ids)
    if args.load_path_i:
        log.info(f'Loading checkpoint from {args.load_path_i}...')
        intensive_model, intensive_step = util.load_model(intensive_model, args.load_path_i, args.intensive_gpu_ids)
    else:
        intensive_step = 0
    intensive_model = intensive_model.to(intensive_device)
    intensive_model.train()
    intensive_ema = util.EMA(intensive_model, args.ema_decay_i)

    #SEX IS FUN
    retro_device, args.retro_gpu_ids = util.get_available_devices()

    # Get Intensive model
    log.info('Building model...')
    retro_model = RetroTrainer()
    retro_model = nn.DataParallel(retro_model, args.retro_gpu_ids)
    retro_step = 0
    retro_model = retro_model.to(retro_device)
    retro_model.train()
    retro_ema = util.EMA(retro_model, args.ema_decay_retro)

    # Get saver
    sketchy_saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)
    #we need two so that they can keep track of each of their respective values 
    intensive_saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)
    # retro saver saves parameters 
    retro_saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)


    # Get optimizer and scheduler
    sketchy_optimizer = optim.Adadelta(sketchy_model.parameters(), args.lr_s,
                               weight_decay=args.l2_wd_s)
    intensive_optimizer = optim.Adadelta(intensive_model.parameters(), args.lr_i,
                               weight_decay=args.l2_wd_i)
    retro_optimizer = optim.Adadelta(retro_model.parameters(), args.lr_r,
                               weight_decay=args.l2_wd_r)
                               
    sketchy_scheduler = sched.LambdaLR(sketchy_optimizer, lambda s: 1.)  # Constant LR
    intensive_scheduler = sched.LambdaLR(intensive_optimizer, lambda s: 1.)  # Constant LR
    retro_scheduler = sched.LambdaLR(retro_optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    #setup losses
    bceLoss = nn.BCEWithLogitsLoss()
    ceLoss = nn.CrossEntropyLoss()

    def train_model(model_name ,model, optimizer, scheduler, tbx, progress_bar, steps_till_eval, 
                    log, ema, step, others, saver, inputs=(0,0,0,0)):
                others = cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.tto(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                y1, y2 = y1.to(device), y2.to(device)

                if model_name == 'sketchy':
                    yi = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                    log_p1 = None
                    log_p2 = None
                    loss = bceLoss(yi, (y1 == -1))
                    for_retro = yi
                elif model_name == 'intensive':                 
                    yi, log_p1, log_p2 = intensive_model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                    loss = args.alpha_1 * bceLoss(yi, (y1 == -1)) + args.alpha_2 * (ceLoss(log_p1, y1) + ceLoss(log_p2, y2))
                    for_retro = (yi, log_p1, log_p2)
                elif model_name == 'retro':
                    i_ans, s_ans, start_ix, end_ix = inputs
                    log_p1, log_p2 = retro_model(i_ans, s_ans, start_ix, end_ix)
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                    for_retro = None
                else:
                    raise ValueError('invalid model name in code')

                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) 
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val) 
                tbx.add_scalar('train/' + model_name, loss_val, step)
                tbx.add_scalar('train/LR-' + model_name,
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {intensive_step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)

                    saver.save(step, model, results[args.metric_name], device, model_name) 

                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step) 
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,  
                                   split='dev',
                                   num_visuals=args.num_visuals
                    return for_retro


    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    intensive_epoch = intensive_step // len(train_dataset) 
    sketchy_epoch = skethy_step // len(train_dataset)
    sketchy_epoch = 0 #set these to zero when also training RV & TAV weights
    intensive_epoch = 0 
    while intensive_epoch <= args.intensive_num_epochs and sketchy_epoch <= args.sketchy_num_epochs:
        intensive_epoch += 1
        sketchy_epoch += 1
        log.info(f'Starting intensive epoch {intensive_epoch} and sketchy epoch{sketchy_epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                others = (cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids)
                if intensive_epoch <= args.intensive_num_epochs:
                   yi_i, log_p1, log_p2 = train_model('intensive' , intensive_model, intensive_optimizer, intensive_scheduler, 
                                tbx, progress_bar, steps_till_eval, log, intensive_ema, intensive_step, others, intensive_saver)
                if sketchy_epoch <= args.sketchy_num_epochs:
                    yi_s = train_model('sketchy' , sketchy_model, sketchy_optimizer, sketchy_scheduler, 
                                tbx, progress_bar, steps_till_eval, log, sketchy_ema, sketchy_step, others, sketchy_saver)
                train_model('retro', retro_model, retro_optimizer, retro_scheduler, 
                                tbx, progress_bar, steps_till_eval, log, retro_ema, retro_step, others, retro_saver, (yi_s, yi_i, log_p1, log_p2))      


def evaluate(model_name, model, data_loader, device, eval_file, max_len, use_squad_v2):
    meter = util.AverageMeter() 

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            y1, y2 = y1.to(device), y2.to(device)
            bceLoss = nn.BCEWithLogitsLoss()
            ceLoss = nn.CrossEntropyLoss()
            if model_name == 'sketchy':
                yi = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss = bceLoss(yi, (y1 == -1)) 
                meter.update(loss.item(), batch_size)
                starts = [0 for x in range(len(ids)-1)]
                starts = starts.insert(0, 1)
                ends= [0 for x in range(len(ids)-1)]
                ends = ends.insert(0, 1)
            elif model_name == 'intensive':                 
                yi, log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss = args.alpha_1 * bceLoss(yi, (y1 == -1)) + args.alpha_2 * (ceLoss(log_p1, y1) + ceLoss(log_p2, y2))
                meter.update(loss.item(), batch_size)
                # Get F1 and EM scores
                p1, p2 = log_p1.exp(), log_p2.exp()
                starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
            else:
                log_p1, log_p2 = retro_model(i_ans, s_ans, start_ix, end_ix)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                meter.update(loss.item(), batch_size)
                p1, p2 = log_p1.exp(), log_p2.exp()
                starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)


            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss_calc=meter.avg)

            # Give us the
            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [(model_name + 'model', meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
