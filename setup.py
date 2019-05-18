"""Build and preprocess the datasets from the scraped files and Download the models' pre-trained weights

Usage:
    Do once:
    > conda env create -f environment.yml
    Then:
    > source activate deepclimb
    > python setup.py

Authors:
    Gael Colas
"""

from args import get_setup_args
from data import binary_preprocess, dataset_split, image_preprocess

if __name__ == '__main__':
    # get command-line args
    args = get_setup_args()

    # pre-process the scraped data into the binary representation
    binary_preprocess.main(args.scraped_data_dir, args.binary_data_dir, args.data_filenames, args.MB_versions, args.grades)
    
    # split the dataset in train/val/test
    split_dict = {"test": args.test_split, "val": args.dev_split, "train": args.train_split}
    dataset_split.main(args.binary_data_dir, args.MB_versions, args.data_filenames, split_dict)
    
    # convert the binary representation into an image representation
    image_preprocess.main(args.scraped_data_dir, args.binary_data_dir, args.image_data_dir, list(split_dict.keys()), args.MB_versions)
    
    # download the models' pre-trained weights
    #download(args_)