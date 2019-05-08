"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute:  provided starter code
    Gael Colas
"""

import argparse

# filenames of the files describing the data
DATA_FILENAMES = ["X", "X_type", "y", "y_user"]

def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Build the datasets and download the model weights')

    add_common_args(parser)

    # directory where the scraped data is stored
    parser.add_argument('--scraped_data_dir',
                        type=str,
                        default="./data/raw",
                        help="If ./data/raw is empty, run > python scraper.py")
    parser.add_argument('--data_filenames',
                        type=list,
                        default=DATA_FILENAMES,
                        help="Filenames of the useful datafiles")
    parser.add_argument('--train_split',
                        type=float,
                        default=0.8,
                        help="Fraction of the dataset to put in the train-set")
    parser.add_argument('--dev_split',
                        type=float,
                        default=0.1,
                        help="Fraction of the dataset to put in the dev-set")    
    parser.add_argument('--test_split',
                        type=float,
                        default=0.1,
                        help="Fraction of the dataset to put in the test-set")
    # TODO: add url of pretrained models' weights

    args = parser.parse_args()

    return args


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'
                         .format(args.metric_name))

    return args


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    # directory where the binary datasets are stored
    parser.add_argument('--binary_data_dir',
                        type=str,
                        default="./data/binary")
    # directory where the image datasets are stored
    parser.add_argument('--image_data_dir',
                        type=str,
                        default="./data/image")
    # versions of the MoonBoard handled
    parser.add_argument('--MB_versions',
                        type=list,
                        default=["2016", "2017"],
                        help="if you want to handle different versions of the MoonBoard, you need to adapt the scraping script and run > python scraper.py")
    # problems' grades considered
    parser.add_argument('--grades',
                        type=tuple,
                        default=('6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+'),
                        help="if you want to handle other grades of the MoonBoard, you need to adapt the scraping script and run > python scraper.py")


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
