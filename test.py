"""Test a model and generate the array of predicted labels.
Generate visualization.

Usage:
    > python test.py --test_split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "val" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run
    
Visualization:
    Saliency maps for 10 random inputs.

Authors:
    Chris Chute (CS224n teaching staff)
        starter code from: https://github.com/chrischute/squad
    Gael Colas
"""

import os
import util
from args import get_test_args
from tqdm import tqdm
from json import dumps
from ujson import load as json_load
import numpy as np
import csv

# to handle images
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter

from models.data_loader import fetch_dataloader
from models.CNN_models import BinaryClimbCNN, ImageClimbCNN, ImageClimbSmallCNN
    

def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get hold embeddings # TODO
    #log.info('Loading embeddings...')
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
    
    log.info('Loading checkpoint from {}...'.format(args.load_path))
    model = util.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    
    # push model on GPU
    model = model.to(device)
    model.eval() # evaluate model

    # Get data loader
    log.info('Building dataset...')
    data_loaders, n_examples = fetch_dataloader([args.test_split], args)
    data_loader = data_loaders[args.test_split]

    # Evaluate
    log.info('Evaluating on {} split...'.format(args.test_split))
    
    # NLL average
    nll_meter = util.AverageMeter()

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

    results = util.evaluate_preds(y_true, y_pred)
    results_list = [('NLL', nll_meter.avg),
                    ('Acc', results['Acc']),
                    ('F1', results['F1']),
                    ('MAE', results['MAE'])]

    results = dict(results_list)

    # Log to console
    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                            for k, v in results.items())
    log.info('{} {}'.format(args.test_split.title(), results_str))

    # Log to TensorBoard
    tbx = SummaryWriter(args.save_dir)
    # visualize examples in Tensorboard
    util.visualize(tbx, y_pred, 0, args.test_split, args.num_visuals, data_loader)
            
    # Write prediction file
    sub_path = os.path.join(args.save_dir, args.test_split + '_' + args.pred_file)
    log.info('Writing submission file to {}...'.format(sub_path))
    np.save(os.path.join(args.save_dir, args.test_split + '_true'), y_true)
    np.save(sub_path, y_pred)
#    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
#        csv_writer = csv.writer(csv_fh, delimiter=',')
#        csv_writer.writerow(['Id', 'Predicted'])
#        for uuid, y_c in enumerate(y_pred):
#            csv_writer.writerow([uuid, y_c])

    # visualize saliency maps
    visualize_saliency(model, data_loader, device, args.save_dir, args.test_split, args.num_visuals, n_examples[args.test_split])

def visualize_saliency(model, data_loader, device, save_dir, split, num_visuals, n_examples):
    """Save saliency maps of image examples.

    Args:
        model (torch.nn.DataParallel): CNN model to use for the visualization.
        data_loader (DataLoader): DataLoader object for the given split.
        device (torch.device): Main device (GPU 0 or CPU).
        save_dir (str): Which folder to store the visualizations.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
        n_examples (int): Number of examples in "split".
    """
    if num_visuals <= 0:
        return
    if num_visuals > n_examples:
        num_visuals = n_examples
        
    # sample 'num_visuals' random examples from 'split' for visualization
    visual_ids = np.random.choice(list(range(n_examples)), size=num_visuals, replace=False)   
        
    # build a batch of random examples
        # unnormalized examples for visualization
    x_visual = torch.stack([data_loader.dataset.__getitem__(idx, visualize=True)[0] for idx in visual_ids])
        # normalized examples to compute the saliency maps
    x_leaf = torch.stack([data_loader.dataset[idx][0] for idx in visual_ids])
        # labels
    y = torch.LongTensor([data_loader.dataset[idx][1] for idx in visual_ids])
        
    # get batch of examples
    #x, y = next(iter(data_loader))
    #x_leaf = x[:num_visuals]
    #y = y[:num_visuals]
        
    with torch.enable_grad():
        # Setup for forward
        x_leaf.requires_grad = True
        x = x_leaf.to(device)
        y = y.to(device)
        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=None, reduction='mean')
        # Backward
        loss.backward()

        # get pixel saliency maps: maximum of absolute value of channels
        saliency_maps = torch.max(torch.abs(x_leaf.grad), (1))[0].numpy()
        
        # get predicted classes
        y = y.cpu().numpy()
        y_pred = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # change to (B, H, W, C)-convention
    x_visual = np.moveaxis(x_visual.numpy(), 1, -1)
    
    # save the saliency maps
    save_path = os.path.join(save_dir, "saliency")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for k in range(num_visuals):
        # saliency map filename
        map_name = "{}_{}_{}_map.jpg".format(k, split, y_pred[k])
        # saliency map
        x_map = saliency_maps[k]
        # normalize the saliency map
        x_map = x_map / np.max(x_map)
        # convert to an image
        map_im = Image.fromarray((x_map*255).astype('uint8'))
        # save to a JPG file
        map_im.save(os.path.join(save_path, map_name), "JPEG")  

        # save the original image for comparison
        im_name = "{}_{}_{}.jpg".format(k, split, y[k])
        x_im = Image.fromarray((x_visual[k]*255).astype('uint8'))
        x_im.save(os.path.join(save_path, im_name), "JPEG") 
        


if __name__ == '__main__':
    main(get_test_args())
