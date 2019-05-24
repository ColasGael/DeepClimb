"""Train a model.

Authors:
    Chris Chute (CS224n teaching staff)
        starter code from: https://github.com/chrischute/squad
    Gael Colas
"""

import util
from args import get_train_args
from tqdm import tqdm
from json import dumps
from ujson import load as json_load

import numpy as np
import random
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from tensorboardX import SummaryWriter

from models.data_loader import fetch_dataloader
from models.CNN_models import BinaryClimbCNN, ImageClimbCNN, ImageClimbSmallCNN

from PIL import Image

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
    
    # Number of classes
    n_classes = len(args.grades)
    
    # Choose model
    log.info('Building model {}...'.format(args.name))

    if 'BinaryClimbCNN' in args.name:
        model = BinaryClimbCNN(n_classes) 
    elif 'ImageClimbCNN' in args.name:
        model = ImageClimbCNN(n_classes)
    elif 'ImageClimbSmallCNN' in args.name:
        model = ImageClimbSmallCNN(n_classes)
    else:
        raise NameError('No model named ' + args.name)
    
    # put model on GPUs
    model = nn.DataParallel(model, args.gpu_ids)
    
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    
    # push model on GPU
    model = model.to(device)
    model.train() # train model

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    data_loaders, n_examples = fetch_dataloader([args.train_split, args.val_split], args)
    train_loader = data_loaders[args.train_split]
    val_loader = data_loaders[args.val_split]
    
    # Train
    log.info('Training on {}-set composed of {} examples...'.format(args.train_split, n_examples[args.train_split]))
    epochs_till_eval = args.eval_epochs
    epoch = step // n_examples[args.train_split]
    while epoch < args.num_epochs:
        epoch += 1
        log.info('Starting epoch {} on {}-set...'.format(epoch, args.train_split))
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for x, y in train_loader: # get batch
                new_im = Image.fromarray(np.moveaxis(np.array(x[0].tolist()), 0, -1), 'RGB')
                new_im.save("test.jpg")
                
                # Setup for forward
                x = x.to(device)

                batch_size = x.size(0)
                optimizer.zero_grad()

                # Forward
                logits = model(x)
                
                # cross-entropy from logits loss
                y = y.to(device)
                loss = F.cross_entropy(logits, y, weight=None, reduction='mean')
                
                # loss value
                loss_val = loss.item()

                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info to TensorBoard
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
        
        epochs_till_eval -= 1
        if epochs_till_eval <= 0:
            epochs_till_eval = args.eval_epochs

            # Evaluate and save checkpoint
            log.info('Evaluating on {}-set at epoch {}...'.format(args.val_split, epoch))
            results, y_pred = evaluate(model, val_loader, device,
                                          args.name,
                                          args.gpu_ids)
            saver.save(step, model, results[args.metric_name], device)

            # Log to console
            results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                    for k, v in results.items())
            log.info('Val {}'.format(results_str))

            # Log to TensorBoard
            log.info('Visualizing in TensorBoard...')
            for k, v in results.items():
                tbx.add_scalar('val/{}'.format(k), v, step)
            #util.visualize(tbx, pred_dict=pred_dict, step=step, split='val', num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, model_name, gpu_ids):
    # NLL average
    nll_meter = util.AverageMeter()

    # put model in eval mode: no gradient computed
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for x, y in data_loader: # get batch           
            # Setup for forward
            x = x.to(device)

            batch_size = x.size(0)

            # Forward
            logits = model(x)

            # cross-entropy from logits loss
            y = y.to(device)
            loss = F.cross_entropy(logits, y, weight=None, reduction='mean')
            nll_meter.update(loss.item(), batch_size)

            # get predicted class
            y_true = y_true + y.tolist()
            y_pred = y_pred + torch.argmax(logits, dim=-1).tolist()

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)


    # put back in train mode
    model.train()

    results = util.evaluate_preds(y_true, y_pred)
    results_list = [('NLL', nll_meter.avg),
                    ('Acc', results['Acc']),
                    ('F1', results['F1']),
                    ('MAE', results['MAE'])]

    results = dict(results_list)

    return results, y_pred


if __name__ == '__main__':
    main(get_train_args())
