#!/usr/bin/env python3

"""Split the preprocessed datasets into Train / Validation and Test sets

The split is random.
But to preserve the overall grades distribution, the split is done per class independently.

Default split: 80%/10%/10%

Authors:
    Gael Colas
"""

# PACKAGES
# to interact with file and folders
import os 
# to handle the matrix representations of examples
import numpy as np
# to fix the seed
from random import seed
# progress bar
from tqdm import trange

def split_trainvaltest(data, split_frac):
    """Split random shuffled data into splits of size given by 'split'
    
    Args:
        'data' (np.array, shape=(n_examples,?): random shuffled data
        'split_frac' (tuple, len=k): indicate the size of the splits
            split_frac[i] = fraction of 'data' to put in the i-th split
    
    Return:
        'data_splits' (list of np.array): the corresponding k splits
        
    Remarks:
        All the fraction of 'split_frac' should sum to one: we want to use the whole dataset.
    """
    assert (sum(split_frac) == 1), "The splits' fractions do not sum to 1!"
    
    # number of examples
    n_examples = data.shape[0]
    
    # number of examples per split
    n_per_split = [int(np.ceil(s*n_examples)) for s in split_frac]
    # indices to split on
    split_idx = np.cumsum(n_per_split)
    split_idx = split_idx[:-1]
    
    # split
    data_splits = np.split(data, split_idx)
            
    return data_splits

def split_sanity_check(ppDirName, VERSIONS, filenames_in, split_names):
    """Check that the split is valid
    
    Args:
        'ppDirName' (String): path to the directory where the preprocessed files are stored
        'VERSIONS' (list of String): versions of the MoonBoard handled
        'filenames_in' (list of String): list of preprocessed files to split
    
    Remark:
        For the split to be valid: if you concatenate all the splits, this new dataset should be a permutation of the old one (dataset before splitting).
    """
    
    def isPermutation(A, B):
        """
        Check if two arrays are permutation of each other
        
        Args:
            'A' (np.array)
            'B' (np.array)
            
        Return:
            'isPermutation' = True if 'A' and 'B' are permutations of each other.
        """
        # check that the arrays have the same shape
        if not (A.shape == B.shape):
            return False
        
        isPermutation = True
        # loop over the rows in A
        for i in trange(A.shape[0]):
            # check that each row of 'A' occurs the same number of time in 'B'
            n_occ_A = np.sum((A[i,:] == A).all(axis=1))
            n_occ_B = np.sum((A[i,:] == B).all(axis=1))
            
            isPermutation = (n_occ_A == n_occ_B)
            
            if not isPermutation:
                return isPermutation

        return isPermutation
    
    for MBversion in VERSIONS:
        print("{:-^100}".format("---Sanity check: split for MoonBoard version {}---".format(MBversion)))

        # path to preprocessed files
        path_in = os.path.join(ppDirName, MBversion)
        
        # old dataset: before splitting
        data_old = []
        
        # new dataset: after splitting
        data_new = []
        
        for filename in filenames_in:
            # load the old dataset
            data = np.load(os.path.join(path_in, filename + ".npy"))
            data_old.append(np.reshape(data, (data.shape[0], -1)))
            
            # load the new dataset splits
            data_splits = []
            for k, split_name in enumerate(split_names):
                filename_out = filename + "_" + split_name + ".npy"
                data_split = np.load(os.path.join(path_in, filename_out))
                data_splits.append(np.reshape(data_split, (data_split.shape[0], -1)))
            
            # build the new dataset: concatenation of the splits
            data_new.append(np.vstack(data_splits))
        
        # concatenate all the dataset files per examples
        dataset_old = np.hstack(data_old)
        dataset_new = np.hstack(data_new)
        
        # check that the split is valid
        assert isPermutation(dataset_old, dataset_new), "The split is not valid: we cannot rebuild the old dataset from the splits!"
    
    print("The splits are valid!")
    
def main(ppDirName, VERSIONS, filenames_in, split_dict):
    # fix the random seed
    seed(1)
    np.random.seed(1)
    
    print("\n DATASETS SPLITS\n")

    for MBversion in VERSIONS:
        print("{:-^100}".format("---Dataset split for MoonBoard version {}---".format(MBversion)))

        # path to preprocessed files
        path_in = os.path.join(ppDirName, MBversion)
        
        # load the label vector 'y'
        try:
            y = np.load(os.path.join(path_in, "y.npy"))
        except Exception as e:
            print("The label file 'y' does not exist: run 'preprocess.py' before and check that the path to the processed files in 'args.py' is correct.")
            raise e
        
        # random shuffling of the dataset (before splitting)
        perm_before = None 
        
        # random shuffling of the dataset (after splitting)
        perm_after = None
        
        first_pass = True
        for filename in filenames_in:
            # load the dataset
            try:
                data = np.load(os.path.join(path_in, filename + ".npy"))
            except Exception as e:
                print("One of the files does not exist: run 'preprocess.py' before and check that the path to the processed files in 'args.py' is correct.")
                raise e
            
            # convert to 2D arrays
            data = np.reshape(data, (y.shape[0], -1))
            
            # random shuffling: same permutation for all the files
            if first_pass:
                perm_before = np.random.permutation(data.shape[0])
            data = data[perm_before,:]   
            
            # train/val/test splits
            data_splits_list = []
            # split per class to preserve the class distribution
            for c in set(y):
                # subset of examples of the given class
                data_subset = data[y[perm_before] == c,:]
                
                # split into train/val/test
                data_splits_y = split_trainvaltest(data_subset, split_dict.values())
                
                # save the splits
                data_splits_list.append(data_splits_y)
                
            # build the splits: merge the per class splits
            data_splits = list(map(lambda x: np.vstack(x), list(zip(*data_splits_list))))
            
            # create one shuffling permutation for all the files
            if first_pass:
                # create the permutation only once
                perm_after = [np.random.permutation(data_split.shape[0]) for data_split in data_splits]
            
            # random shuffling of the dataset (after splitting)
            data_splits = [data_split[perm_after[i],:] for i, data_split in enumerate(data_splits)]

            # save the splits
            for k, split_name in enumerate(split_dict.keys()):
                filename_out = filename + "_" + split_name
                np.save(os.path.join(path_in, filename_out), data_splits[k].squeeze())
                
                if first_pass:
                    print("There are {} examples in the {} set: {:.2f}% of the whole dataset.".format(data_splits[k].shape[0], split_name, data_splits[k].shape[0]/data.shape[0]))
            
            first_pass = False
        
if __name__ ==  "__main__":
    # versions of the MoonBoard handled
    VERSIONS = ["2016", "2017"]
    # directory where the preprocessed files are stored
    ppDirName = 'binary'
    # files in the dataset to split
    filenames_in = ["X", "X_type", "y", "y_user"]
    # dataset split
    split_dict = {"test": 0.1, "val": 0.1, "train": 0.8}
    
    main(ppDirName, VERSIONS, filenames_in, split_dict)    
    
    run_sanity_check = False
    if run_sanity_check:
        split_sanity_check(ppDirName, VERSIONS, filenames_in, split_dict.keys())