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
from models import SketchyReader
from models import IntensiveReader
from models import RetroQANet
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

# TO TRAIN YOU MUST ALSO SET --model_name (skecthy or intensive)


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

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

    # Get model
    log.info('Building model...')
    if args.model_name == 'sketchy':
        model = SketchyReader(word_vectors=word_vectors,
                              char_vectors=char_vectors,
                              hidden_size=args.hidden_size,
                              drop_prob=args.drop_prob)  # SKETCHY
    elif args.model_name == 'intensive':

        model = IntensiveReader(word_vectors=word_vectors,
                                char_vectors=char_vectors,
                                hidden_size=args.hidden_size,
                                drop_prob=args.drop_prob)  # INTENSIVE
    elif args.model_name == 'retro':

        model = RetroQANet(word_vectors=word_vectors,
                            char_vectors=char_vectors,
                            hidden_size=args.hidden_size,
                            intensive_path=args.load_path_i,
                            sketchy_path=args.load_path_s,
                            gpu_ids=args.gpu_ids,
                            drop_prob=args.drop_prob) #Outer

    model = nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # setup losses
    bceLoss = nn.BCELoss()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

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
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                y1, y2 = y1.to(device), y2.to(device)
                if args.model_name == 'sketchy':
                    yi = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                    log_p1 = None
                    log_p2 = None
                    loss = bceLoss(yi, torch.where(y1 == 0, 1, 0).type_as(yi))
                elif args.model_name == 'intensive':
                    yi, log_p1, log_p2 = model(
                        cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                    loss = args.alpha_1 * bceLoss(yi, torch.where(y1 == 0, 1, 0).type_as(
                        yi)) + args.alpha_2 * (F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2))
                elif args.model_name == 'retro':
                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                else:
                    raise ValueError(
                        'invalid --model_name, sketchy or intensive required')

                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/' + args.model_name, loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2,
                                                  model_name=args.model_name, 
                                                  a1=args.alpha_1, 
                                                  a2=args.alpha_2)
                    saver.save(
                        step, model, results[args.metric_name], device, model_name=args.model_name)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(
                        f'{k}: {v:05.2f}' for k, v in results.items())
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
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2, model_name="", a1=0.1, a2=0.9):
    meter = util.AverageMeter()

    # setup losses
    bceLoss = nn.BCELoss()
    ceLoss = nn.CrossEntropyLoss()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            y1, y2 = y1.to(device), y2.to(device)
            if model_name == 'sketchy':
                yi = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                loss = bceLoss(yi, torch.where(y1 == 0, 1, 0).type_as(yi))
                starts, ends = [y1, y2]
            elif model_name == 'intensive':
                yi, log_p1, log_p2 = model(
                    cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                loss = a1 * bceLoss(yi, torch.where(y1 == 0, 1, 0).type_as(
                    yi)) + a2 * (F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2))
                # Get F1 and EM scores
                p1, p2 = log_p1.exp(), log_p2.exp()
                starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
                starts, ends = starts.tolist(), ends.tolist()
            elif model_name == 'retro':
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                p1, p2 = log_p1.exp(), log_p2.exp()
                starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
                starts, ends = starts.tolist(), ends.tolist()
            else:
                raise ValueError(
                    'invalid --model_name, sketchy or intensive required')
            meter.update(loss.item(), batch_size)
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss_calc=meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts,
                                           ends,
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('Loss', meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
