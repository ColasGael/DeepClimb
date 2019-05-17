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

from args import get_train_args
from json import dumps
from CNN_models import BinaryClimbCNN, ImageClimbCNN
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

import util

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids)) # scale batch size to number of available GPUs

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get hold embeddings # TODO
    log.info('Loading embeddings...')
    #word_vectors = util.torch_from_json(args.word_emb_file)

    # Choose model
    log.info('Building model {}...'.format(args.name))

    if 'baseline' in args.name:
        model = BiDAF(word_vectors=word_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob)

    if args.name == 'BiDAF_char':
        model = BiDAF_char(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob)

    elif (args.name == 'BiDAF_tag') or (args.name == 'BiDAF_tag_loss'):
        model = BiDAF_tag(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob)

    elif (args.name == 'BiDAF_tag_unfrozen') or (args.name == 'BiDAF_tag_unfrozen_loss'):
        model = BiDAF_tag(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob,
                      freeze_tag=False)

    elif args.name == 'BiDAF_tag_ext':
        model = BiDAF_tag_ext(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob)

    elif args.name == 'BiDAF_tag_ext_unfrozen':
        model = BiDAF_tag_ext(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob,
                      freeze_tag=False)

    elif args.name == 'coattn':
        model = CoattentionModel(hidden_dim=args.hidden_size,
                                embedding_matrix=word_vectors,
                                train_word_embeddings=False,
                                dropout=0.35,
                                pooling_size=16,
                                number_of_iters=4,
                                number_of_layers=2,
                                device=device)

    else:
        raise NameError('No model named ' + args.name)

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

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
                                   collate_fn=collate_fn,
                                   drop_last=True)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn,
                                 drop_last=True)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, cpos_idxs, cner_idxs, cw_ems, cw_tfs, qw_idxs, qc_idxs, qpos_idxs, qner_idxs, qw_ems, qw_tfs, y1, y2, ids in train_loader: # NEW
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)

                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                if 'baseline' in args.name:
                    log_p1, log_p2 = model(cw_idxs, qw_idxs)

                elif args.name == 'BiDAF_char':
                    # Additional setup for forward
                    cc_idxs = cc_idxs.to(device)
                    qc_idxs = qc_idxs.to(device)
                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)

                elif (args.name == 'BiDAF_tag') or (args.name == 'BiDAF_tag_unfrozen') or (args.name == 'BiDAF_tag_loss') or (args.name == 'BiDAF_tag_unfrozen_loss'):
                    # Additional setup for forward
                    cc_idxs = cc_idxs.to(device)
                    cpos_idxs = cpos_idxs.to(device)
                    cner_idxs = cner_idxs.to(device)
                    qc_idxs = qc_idxs.to(device)
                    qpos_idxs = qpos_idxs.to(device)
                    qner_idxs = qner_idxs.to(device)

                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs)

                elif (args.name == 'BiDAF_tag_ext') or (args.name == 'BiDAF_tag_ext_unfrozen'):
                    # Additional setup for forward
                    cc_idxs = cc_idxs.to(device)
                    cpos_idxs = cpos_idxs.to(device)
                    cner_idxs = cner_idxs.to(device)
                    cw_ems = cw_ems.to(device)
                    cw_tfs = cw_tfs.to(device)
                    qc_idxs = qc_idxs.to(device)
                    qpos_idxs = qpos_idxs.to(device)
                    qner_idxs = qner_idxs.to(device)
                    qw_ems = qw_ems.to(device)
                    qw_tfs = qw_tfs.to(device)

                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs, cw_ems, qw_ems, cw_tfs, qw_tfs)

                elif args.name == 'coattn':
                    max_c_len = cw_idxs.size(1)
                    max_q_len = qw_idxs.size(1)

                    c_len = []
                    q_len = []

                    for i in range(cw_idxs.size(0)):
                        if len((cw_idxs[i] == 0).nonzero()) != 0:
                            c_len_i = (cw_idxs[i] == 0).nonzero()[0].item()
                        else:
                            c_len_i = cw_idxs.size(1)

                        if len((qw_idxs[i] == 0).nonzero()) != 0:
                            q_len_i = (qw_idxs[i] == 0).nonzero()[0].item()
                        else:
                            q_len_i = qw_idxs.size(1)

                        c_len.append(int(c_len_i))
                        q_len.append(int(q_len_i))

                    c_len = torch.Tensor(c_len).int()
                    q_len = torch.Tensor(q_len).int()

                    num_examples = int(cw_idxs.size(0) / len(args.gpu_ids))

                    log_p1, log_p2 = model(max_c_len, max_q_len, cw_idxs, qw_idxs, c_len, q_len, num_examples, True, True)

                else:
                    raise NameError('No model named ' + args.name)

                y1, y2 = y1.to(device), y2.to(device)

                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)

                # Add distance penalization
                if 'loss' in args.name:
                    loss += distance_criterion(log_p1, y1) + distance_criterion(log_p2, y2)

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
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2,
                                                  args.name,
                                                  args.gpu_ids)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2, model_name, gpu_ids):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, cpos_idxs, cner_idxs, cw_ems, cw_tfs, qw_idxs, qc_idxs, qpos_idxs, qner_idxs, qw_ems, qw_tfs, y1, y2, ids in data_loader: # NEW
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if 'baseline' in model_name:
                log_p1, log_p2 = model(cw_idxs, qw_idxs)

            elif model_name == 'BiDAF_char':
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)

            elif (model_name == 'BiDAF_tag') or (model_name == 'BiDAF_tag_unfrozen') or (model_name == 'BiDAF_tag_loss') or (model_name == 'BiDAF_tag_unfrozen_loss'):
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                cpos_idxs = cpos_idxs.to(device)
                cner_idxs = cner_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                qpos_idxs = qpos_idxs.to(device)
                qner_idxs = qner_idxs.to(device)

                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs)

            elif (model_name == 'BiDAF_tag_ext') or (model_name == 'BiDAF_tag_ext_unfrozen'):
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                cpos_idxs = cpos_idxs.to(device)
                cner_idxs = cner_idxs.to(device)
                cw_ems = cw_ems.to(device)
                cw_tfs = cw_tfs.to(device)
                qc_idxs = qc_idxs.to(device)
                qpos_idxs = qpos_idxs.to(device)
                qner_idxs = qner_idxs.to(device)
                qw_ems = qw_ems.to(device)
                qw_tfs = qw_tfs.to(device)

                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs, cw_ems, qw_ems, cw_tfs, qw_tfs)

            elif args.name == 'coattn':
                max_c_len = cw_idxs.size(1)
                max_q_len = qw_idxs.size(1)

                c_len = []
                q_len = []

                for i in range(cw_idxs.size(0)):
                    if len((cw_idxs[i] == 0).nonzero()) != 0:
                        c_len_i = (cw_idxs[i] == 0).nonzero()[0].item()
                    else:
                        c_len_i = cw_idxs.size(1)

                    if len((qw_idxs[i] == 0).nonzero()) != 0:
                        q_len_i = (qw_idxs[i] == 0).nonzero()[0].item()
                    else:
                        q_len_i = qw_idxs.size(1)

                    c_len.append(int(c_len_i))
                    q_len.append(int(q_len_i))

                c_len = torch.Tensor(c_len).int()
                q_len = torch.Tensor(q_len).int()

                num_examples = int(cw_idxs.size(0) / len(gpu_ids))

                log_p1, log_p2 = model(max_c_len, max_q_len, cw_idxs, qw_idxs, c_len, q_len, num_examples, True, False)

            else: # default: run baseline
                log_p1, log_p2 = model(cw_idxs, qw_idxs)

            y1, y2 = y1.to(device), y2.to(device)
            #if model_name == 'coattn':
            #    loss = nn.CrossEntropyLoss()(log_p1, y1) + nn.CrossEntropyLoss()(log_p2, y2)
            #else:
            #    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            #if model_name != 'coattn':
            #    p1, p2 = log_p1.exp(), log_p2.exp()
            #else:
            #    p1, p2 = log_p1, log_p2
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
