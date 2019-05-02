"""Build and preprocess the datasets from the scraped files and Download the models' pre-trained weights

Usage:
    > source activate deepclimb
    > python setup.py

Authors:
    Gael Colas
"""

from args import get_setup_args
from data import binary_preprocess, dataset_split

if __name__ == '__main__':
    # get command-line args
    args = get_setup_args()

    # pre-process the scraped data
    binary_preprocess.main(args.scraped_data_dir, args.binary_data_dir, args.data_filenames, args.MB_versions, args.grades)
    
    # split the dataset in train/dev/test
    split_dict = {"test": args.test_split, "dev": args.dev_split, "train": args.train_split}
    dataset_split.main(args.binary_data_dir, args.MB_versions, args.data_filenames, split_dict)
    
    # download the models' pre-trained weights
    #download(args_)