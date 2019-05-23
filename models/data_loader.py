"""Define the CNN Dataset classes and the method to fetch the corresponding DataLoaders.

Available Datasets:
    - 'BinaryDataset' : an example is represented as a binary matrix (indicating the available holds for a problem).
Version: MoonBoard 2016 only.
Transformation: None
Data augmentation: None.

    - 'ImageDataset' : an example is represented as an image (where the available holds are circled on the MoonBoard template image).
Version: MoonBoard 2016 and 2017.
Transformation: resize to (H, W) = (384, 256)
Data augmentation: flip left-right

Authors:
    Surag Nair (CS230 teaching staff)
        starter code from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py
    Gael Colas
"""

import random
import os

import numpy as np
from PIL import Image
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# MoonBoard grid properties
GRID_DIMS = (18, 11) # dimensions

# TRAIN image transformation pipeline 
train_transformer = transforms.Compose([
    transforms.Resize(256),                                         # resize to (393, 256) 
    transforms.Lambda(lambda img: img.crop(box=(0, 0, 256, 384))),  # crop to (384, 256)
    transforms.RandomHorizontalFlip(),                              # randomly flip image horizontally
    transforms.ToTensor()])                                         # transform it into a torch tensor

# EVAL image transformation pipeline (no horizontal flip)
eval_transformer = transforms.Compose([
    transforms.Resize(256),                                         # resize to (393, 256) 
    transforms.Lambda(lambda img: img.crop(box=(0, 0, 256, 384))),  # crop to (384, 256)
    transforms.ToTensor()])                                         # transform it into a torch tensor

    
class ClimbBinaryDataset(Dataset):
    """PyTorch definition of Dataset to deal with the binary representation of examples.
    
    Attributes:
        'X' (np.array, ): filenames of the dataset's examples
        'y' (list of int): corresponding examples' labels
    """
    
    def __init__(self, data_dir, split, n_dev=3*64):
        """Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        
        Args:
            'data_dir' (String): directory containing the dataset
            'split' (String): dataset split to load
                            Possible choices: 'train', 'dev', 'val' and 'test'
            'n_dev' (int, default=3*64): number of train examples to include in development set
        
        Remark:
            'n_dev' should be a multiple of the batch size.                            
        """
        
        # load the data
        if split == "dev":
            self.X = np.load(os.path.join(data_dir, "X_{}".format("train")))
            self.y = np.load(os.path.join(data_dir, "y_{}".format("train")))
            
            # crop the number of examples for the development set
            self.X = self.X[:n_dev] # shape (None, prod(GRID_DIMS))
            self.y = self.y[:n_dev]
            
        else:
            self.X = np.load(os.path.join(data_dir, "X_{}".format(split)))
            self.y = np.load(os.path.join(data_dir, "y_{}".format(split)))
            
        # reshape the examples in grid form
        self.X = np.reshape(self.X, (-1, GRID_DIMS))
        print(self.X.shape, self.y.shape, self.y.size)
        
    def __len__(self):
        """Return the size of dataset = number of distinct examples.
        """
        return self.y.size

    def __getitem__(self, idx):
        """Get example pair (image, label) from index. Perform transforms on image.
        
        Args:
            'idx' (int): index in [0, 1, ..., size_of_dataset-1]
            
        Returns:
            'x' (torch.Tensor, shape=GRID_DIMS, dtype=torch.int64): binary matrix of the example
            'y' (int): corresponding label of image
        """
        
        x = from_numpy(self.X[idx]).long() 
        
        return x, self.y[idx]

class ClimbImageDataset(Dataset):
    """PyTorch definition of Dataset to deal with the image representation of examples.
    
    Attributes:
        'filenames' (list of String): filenames of the dataset's examples
        'y' (list of int): corresponding examples' labels
        'transform' (torchvision.transforms): transformation to apply on image
    """
    
    def __init__(self, data_dirs, split, transform, n_dev=3*64):
        """Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        
        Args:
            'data_dirs' (list of String): directories containing the dataset
            'split' (String): dataset split to load
                            Possible choices: 'train', 'dev', 'val' and 'test'
            'transform' (torchvision.transforms): transformation to apply on image    
            'n_dev' (int, default=3*64): number of train examples to include in development set
        
        Remark:
            'n_dev' should be a multiple of the batch size.
        """
        # initialization
        self.filenames = []
        self.labels = []
        
        for data_dir in data_dirs:
            # get the path to the files in the split
            if split == "dev":
                data_path = os.path.join(data_dir, "train")
            else:
                data_path = os.path.join(data_dir, split)
            
            # get list of files in the directory (filter only the .jpg files)
            self.filenames = self.filenames + [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
            
        # clip the number of examples for the development set
        if split == "dev":
            self.filenames = self.filenames[:n_dev]
        
        # get the class label: filename = '<label>_<split>_<example_nb>.jpg'
        self.y = [int(os.path.split(filename)[-1].split('_')[0]) for filename in self.filenames]
        
        self.transform = transform

    def __len__(self):
        """Return the size of dataset = number of distinct examples.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """Get example pair (image, label) from index. Perform transforms on image.
        
        Args:
            'idx' (int): index in [0, 1, ..., size_of_dataset-1]
        Returns:
            'image' (Tensor): transformed image
            'label' (int): corresponding label of image
        """
        
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        
        return image, self.y[idx]


def fetch_dataloader(splits, params):
    """Fetch the DataLoader object for each split in splits from data_dir.
    
    Args:
        'splits' (list of String): has one or more of 'train', 'val', 'test' depending on which data is required
        'params' (Params): hyperparameters
        
    Returns:
        'dataloaders' (dict: str -> DataLoader): DataLoader object for each split in splits
        'dataloaders' (dict: str -> int): number of examples for each split in splits
    """
    dataloaders = {}
    n_examples = {}

    # available MoonBoard versions
    versions = params.MB_versions
    
    # get the paths to the data directories
    if params.use_image: #whether to use the image or the binary representation of examples
        data_dir = params.image_data_dir
        data_paths = [os.path.join(data_dir, version) for version in versions] # use all the available versions
    else: 
        data_dir = params.binary_data_dir
        data_path = os.path.join(data_dir, versions[0]) # only use one version dataset
    
    for split in ['train', 'dev', 'val', 'test']:
        if split in splits:
        
            # use the train_transformer if training data, else use eval_transformer without random flip
            if (split == 'train') or (split == 'dev'):
                transformer = train_transformer
                shuffle = True
            else:
                transformer = eval_transformer
                shuffle = False
            
            if params.use_image: 
                dataset = ClimbImageDataset(data_paths, split, transformer, n_dev=params.batch_size)
                
            else:                
                dataset = ClimbBinaryDataset(data_path, split, n_dev=params.batch_size)
            
            n_examples[split] = len(dataset)
            dataloaders[split] = DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle, num_workers=params.num_workers)

    return dataloaders, n_examples