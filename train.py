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
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids_sketchy)) #What issues may this cause (Alll GPU's dedicated to skethcy?), batch size only works for sketchy and not for 

    #maybe we make batchsize equal to hald of this
    #Take half of available devices?

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get Sketchy model
    log.info('Building model...')
    sketchy_model = #QANET SKETCHY MODEL INTITIATION
    model = nn.DataParallel(sketchy_model, args.sketchy_gpu_ids)
    if args.load_path_s:
        log.info(f'Loading checkpoint from {args.load_path_s}...')
        sketchy_model, sketchy_step = util.load_model(sketchy_model, args.load_path_s, args.sketchy_gpu_ids)
    else:
        sketchy_step = 0
    sketchy_model = sketchy_model.to(sketchy_device)
    sketchy_model.train()
    skettchy_ema = util.EMA(sketchy_model, args.ema_decay_s)
     
    #check again for valid devices
    intensive_device, args.intensive_gpu_ids = util.get_available_devices()

    # Get Intensive model
    log.info('Building model...')
    intensive_model = #QANET INTENSIVE MODEL INTITIATION
    model = nn.DataParallel(intensive_model, args.intensive_gpu_ids)
    if args.load_path_i:
        log.info(f'Loading checkpoint from {args.load_path_i}...')
        intensive_model, intensive_step = util.load_model(intensive_model, args.load_path_i, args.intensive_gpu_ids)
    else:
        intensive_step = 0
    intensive_model = intensive_model.to(intensive_device)
    intensive_model.train()
    intensive_ema = util.EMA(intensive_model, args.ema_decay_i)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    sketchy_optimizer = optim.Adadelta(sketchy_model.parameters(), args.lr_s,
                               weight_decay=args.l2_wd_s)
    intensive_optimizer = optim.Adadelta(intensive_model.parameters(), args.lr_i,
                               weight_decay=args.l2_wd_i)
                               
    sketchy_scheduler = sched.LambdaLR(sketchy_optimizer, lambda s: 1.)  # Constant LR
    intensive_scheduler = sched.LambdaLR(intensive_optimizer, lambda s: 1.)  # Constant LR

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

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = intensive_step // len(train_dataset) #WHAT SHOULD WE DO ABOUT STEP HERE? SEPERATE?
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                sketchy_cw_idxs = cw_idxs.to(sketchy_device)
                sketchy_qw_idxs = qw_idxs.to(sketchy_device)
                intensive_cw_idxs = cw_idxs.to(intensive_device)    #Sending signals sto the seperate devices
                intensive_qw_idxs = qw_idxs.to(intensive_device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                yi_s = sketchy_model(sketchy_cw_idxs, sketchy_qw_idxs)
                yi_i, log_p1, log_p2 = intensive_model(intensive_cs_idxs, intensive_qw_idxs)
                sketchy_y1, sketchy_y2 = y1.to(sketchy_device), y2.to(sketchy_device)
                intensive_y1, intensive_y2 = y1.to(intensive_device), y2.to(intensive_device)
                bceLoss = nn.BCEWithLogitsLoss()
                ceLoss = nn.CrossEntropyLoss()
                sketchy_loss = bceLoss(yi_s, ???) #How do we represnet unanswerable questions? <<<<<<<<<<<<<<<<<<<
                intensive_loss = args.alpha_1 * bceLoss(yi_i, ???) + args.alpha_2 * (ceLoss(log_p1, intensive_y1) + ceLoss(log_p2, intensive_y2))
                #loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                sketchy_loss_val = sketchy_loss.item()
                intensive_loss_val = intensive_loss.item()

                # Backward
                sketchy_loss.backward()
                intensive_loss.backward()
                nn.utils.clip_grad_norm_(sketchy_model.parameters(), args.max_grad_norm) #requires us to have the same grad norm
                nn.utils.clip_grad_norm_(intensive_model.parameters(), args.max_grad_norm)
                sketchy_optimizer.step()
                intensive_optimizer.step()
                sketchy_scheduler.step(sketchy_step // batch_size)
                intensive_scheduler.step(intesnive_step // batch_size)  #Requires us to have the same batch size
                sketchy_ema(sketchy_model, sketchy_step // batch_size)
                intense_ema(intensive_model, intensive_ step // batch_size)

                # Log info
                intensice_step += batch_size #IS STEP USED RIGHT?
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         BCE=sketchy_loss_val, ???) #How can we change this to handle the intensive loss
                tbx.add_scalar('train/Sketchy', skeychy_loss_val, sketchy_step)
                tbx.add_scalar('train/Intensive', intensive_loss_val, intensive_step)
                tbx.add_scalar('train/LR_Intensive',
                               intensive_optimizer.param_groups[0]['lr'],
                               intensive_step)
                tbx.add_scalar('train/LR_Sketchy',
                               sketchy_optimizer.param_groups[0]['lr'],
                               sketchy_step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {intensive_step}...') #This would indicate that the steps for each need to be equal, Look into this
                    ema_sketchy.assign(sketchy_model)
                    ema_intensive.assign(intensive_model)
                    sketchy_results, sketchy_pred_dict = evaluate(sketchy_model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    intensive_results, intensive_pred_dict = evaluate(intensive_model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(sketchy_step, sketchy_model, results[args.metric_name], sketchy_device)
                    saver.save(intensive_step, intensive_model, results[args.metric_name], intensive_device)
                    sketchy_ema.resume(sketchy_model)
                    intensive_ema.resume(intensive_model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step) #Also implies a signular step here... what is the dependency of step - Is training with the same step problematic
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,  #Same issue with step here
                                   split='dev',
                                   num_visuals=args.num_visuals)

#Evaluate Needs to be set to have a proper inference otherwise mostly complete
def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    sketchy_meter = util.AverageMeter() 
    intensive_meter = util.AverageMeter()

    sketchy_model.eval()
    intensive_model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            sketchy_cw_idxs = cw_idxs.to(sketchy_device)
            sketchy_qw_idxs = qw_idxs.to(sketchy_device)
            intensive_cw_idxs = cw_idxs.to(intensive_device)    #Sending signals sto the seperate devices
            intensive_qw_idxs = qw_idxs.to(intensive_device)
            batch_size = cw_idxs.size(0)

            # Forward
            yi_s = sketchy_model(sketchy_cw_idxs, sketchy_qw_idxs)
            yi_i, log_p1, log_p2 = intensive_model(intensive_cs_idxs, intensive_qw_idxs)
            sketchy_y1, sketchy_y2 = y1.to(sketchy_device), y2.to(sketchy_device)
            intensive_y1, intensive_y2 = y1.to(intensive_device), y2.to(intensive_device)
            bceLoss = nn.BCEWithLogitsLoss()
            ceLoss = nn.CrossEntropyLoss()
            sketchy_loss = bceLoss(yi_s, ???) #How do we represnet unanswerable questions? <<<<<<<<<<<<<<<<<<<
            intensive_loss = args.alpha_1 * bceLoss(yi_i, ???) + args.alpha_2 * (ceLoss(log_p1, intensive_y1) + ceLoss(log_p2, intensive_y2))
            #loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            sketchy_meter.update(sketchy_loss.item(), batch_size)
            intensive_meter.update(intensive_loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Sketchy=sketchy_meter.avg, Intensive=intensive_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    sketchy_model.train()
    intensive_model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('Sketchy', sketchy_meter.avg),
                    ('Intensive', intensive_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
