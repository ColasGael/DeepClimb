"""Test a model and generate prediction CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "val" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Authors:
    Chris Chute (chute@stanford.edu)
    Gael Colas
"""

from os.path import join
import util
from args import get_test_args
from tqdm import tqdm
from json import dumps
from ujson import load as json_load
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter

from models.data_loader import data_loader
from models.CNN_models import BinaryClimbCNN, ImageClimbCNN
    

def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get hold embeddings # TODO
    log.info('Loading embeddings...')
    #word_vectors = util.torch_from_json(args.word_emb_file)
    
    # Number of classes
    n_classes = len(args.grades)
    
    # Choose model
    log.info('Building model {}...'.format(args.name))

    if args.name == 'BinaryClimbCNN':
        model = ImageClimbCNN(n_classes) 
    elif args.name == 'ImageClimbCNN':
        model = ImageClimbCNN(n_classes)
    else:
        raise NameError('No model named ' + args.name)
    
    # put model on GPUs
    model = nn.DataParallel(model, args.gpu_ids)
    
    log.info('Loading checkpoint from {}...'.format(args.load_path))
    model, step = util.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    
    # push model on GPU
    model = model.to(device)
    model.eval() # evaluate model

    # Get data loader
    log.info('Building dataset...')
    data_loaders = fetch_dataloaders([args.split], args)
    data_loader = data_loaders[args.split]

    # Evaluate
    log.info('Evaluating on {} split...'.format(args.split))
    
    # NLL average
    nll_meter = util.AverageMeter()

    y_true = []
    y_pred = []

    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for x, y in train_loader: # get batch
            # Setup for forward
            x = x.to(device)

            batch_size = x.size(0)

            # Forward
            logits = model(x)

            y = y.to(device)

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

    results = util.eval_preds(y_true, y_pred)
    results_list = [('NLL', nll_meter.avg),
                    ('Acc', results['Acc']),
                    ('F1', results['F1']),
                    ('MAE', results['MAE'])]

    results = OrderedDict(results_list)

    # Log to console
    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                            for k, v in results.items())
    log.info('{} {}'.format(args.split.title(), results_str))

    # Log to TensorBoard
    tbx = SummaryWriter(args.save_dir)
    #util.visualize(tbx, pred_dict=pred_dict, step=step, split='val', num_visuals=args.num_visuals)

    # Write prediction file
    sub_path = join(args.save_dir, args.split + '_' + args.pred_file)
    log.info('Writing submission file to {}...'.format(sub_path))
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid, y_c in enumerate(y_pred):
            csv_writer.writerow([uuid, y_c])


if __name__ == '__main__':
    main(get_test_args())
