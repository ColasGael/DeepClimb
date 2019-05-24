"""Convolutional Neural Network models

Authors:
    Gael Colas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MoonBoard grid properties
GRID_DIMS = (18, 11) # dimensions

class BinaryClimbCNN(nn.Module):
    """CNN architecture to predict the grade class from the binary representation of examples.
    
    Architecture: 
        [CONV3-n_channels*(2^k) - BN - ReLU - MaxPool2]*3 - [CONV3-n_channels*4 - BN - ReLU - AvgPool(3,2)] - [CONV1-n_channels*8 - BN - ReLU] - [CONV1-n_classes]
    
    Remark:
        binary matrix shape: GRID_DIMS
    """
    
    def __init__(self, n_classes, n_channels=8):
        super(BinaryClimbCNN, self).__init__()
        
        conv1 = nn.Conv2d(1, n_channels, (3,3), padding=1, bias=True)              # (N, n_channels, 18, 11)
        bn1 = nn.BatchNorm2d(n_channels)                                           # (N, n_channels, 18, 11)
        relu1 = nn.ReLU()                                                          # (N, n_channels, 18, 11)
        maxpool1 = nn.MaxPool2d(2, padding=1)                                      # (N, n_channels, 10, 6)
        
        conv2 = nn.Conv2d(n_channels, n_channels*2, (3,3), padding=1, bias=True)   # (N, n_channels*2, 10, 6)
        bn2 = nn.BatchNorm2d(n_channels*2)                                         # (N, n_channels*2, 10, 6)
        relu2 = nn.ReLU()                                                          # (N, n_channels*2, 10, 6)
        maxpool2 = nn.MaxPool2d(2)                                                 # (N, n_channels*2, 5, 3)
        
        conv3 = nn.Conv2d(n_channels*2, n_channels*4, (3,3), padding=1, bias=True) # (N, n_channels*4, 5, 3)
        bn3 = nn.BatchNorm2d(n_channels*4)                                         # (N, n_channels*4, 5, 3)
        relu3 = nn.ReLU()                                                          # (N, n_channels*4, 5, 3)
        maxpool3 = nn.MaxPool2d(2, padding=1)                                      # (N, n_channels*4, 3, 2)
        
        conv4 = nn.Conv2d(n_channels*4, n_channels*8, (3,3), padding=1, bias=True) # (N, n_channels*8, 3, 2)
        bn4 = nn.BatchNorm2d(n_channels*8)                                         # (N, n_channels*8, 3, 2)
        relu4 = nn.ReLU()                                                          # (N, n_channels*8, 3, 2)
        avgpool = nn.AvgPool2d((3, 2))                                             # (N, n_channels*8, 1, 1)
        
        conv5 = nn.Conv2d(n_channels*8, n_channels*8, (1,1), bias=True)            # (N, n_channels*8, 1, 1)
        bn5 = nn.BatchNorm2d(n_channels*8)                                         # (N, n_channels*8, 1, 1)
        relu5 = nn.ReLU()                                                          # (N, n_channels*8, 1, 1)
        
        conv5 = nn.Conv2d(n_channels*8, n_classes, (1,1), bias=True)               # (N, n_classes, 1, 1)
        
        # encapsulation
        self.network = Sequential(self.conv1, self.bn1, self.relu1, self.maxpool1,
                                self.conv2, self.bn2, self.relu2, self.maxpool2,
                                self.conv3, self.bn3, self.relu3, self.maxpool3,
                                self.conv4, self.bn4, self.relu4, self.avgpool,
                                self.conv5, self.bn5, self.relu5, self.conv5)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass for a batch of examples.
        
        Args:
            'x' (torch.Tensor, shape=(N, *GRID_DIMS), dtype=torch.int64): batch of N examples represented as binary matrices
        
        Return:
            'logits' (torch.Tensor, shape=(N, n_classes)): batch of scores for each class
        """
        # create channel dimension
        x = torch.unsqueeze(x, 1)       # (N, 1, 18, 11)
        
        # forward pass through the network
        logits = self.network(x)        # (N, n_classes, 1, 1)
        
        # reshape to get the scores
        logits = torch.squeeze(logits)       # (N, n_classes)
        
        return logits

        
class ImageClimbCNN(nn.Module):
    """CNN architecture to predict the grade class from the image representation of examples.
    
    Architecture: 
        [CONV3-n_channels*(2^(k//2)) - BN - ReLU - MaxPool2]*n_conv_blocks - [CONV3-n_channels*16 - BN - ReLU - AvgPool(2,3)] - [CONV1-n_classes]
    
    Remark:
        Image matrix shape: (C, W, H) = (3, 256, 384)
    """
    
    def __init__(self, n_classes, n_channels=8, n_conv_blocks=8):
        super(ImageClimbCNN, self).__init__()
        
        n_channels_in = 3
        
        # convolutional blocks
        conv_blocks = []
        for k in range(n_conv_blocks):
            n_channels_out = n_channels*2**(k//2)
            conv_blocks.append(self._conv_block(n_channels_in, n_channels_out))
            n_channels_in = n_channels_out
        
        # average pooling
        avgpool_block = self._conv_block(n_channels_in, n_channels_in, pool_method="avg")
        
        # 1x1 final convolution
        conv1 = nn.Conv2d(n_channels_in, n_classes, (1,1), bias=True)
        
        self.network = nn.Sequential(*conv_blocks, avgpool_block, conv1)
            
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass for a batch of examples.
        
        Args:
            'x' (torch.Tensor, shape=(N, 3, 256, 384), dtype=torch.int64): batch of N examples represented as binary matrices
        
        Return:
            'logits' (torch.Tensor, shape=(N, n_classes)): batch of scores for each class
        """        
        # forward pass through the network
        logits = self.network(x)        # (N, n_classes, 1, 1)
        
        # reshape to get the scores
        logits = torch.squeeze(logits)  # (N, n_classes)
        
        return logits
    
    def _conv_block(self, n_channels_in, n_channels_out, filter_size=3, pool_method="max"):
        padding = int((filter_size-1)/2)
    
        conv = nn.Conv2d(n_channels_in, n_channels_out, filter_size, padding=padding, bias=True)            # (N, n_channels_out, W, H)
        bn = nn.BatchNorm2d(n_channels_out, track_running_stats=True, momentum=1.)                                       # (N, n_channels_out, W, H)
        relu = nn.ReLU()                                                                                    # (N, n_channels_out, W, H)
        
        if pool_method == "max":
            pool = nn.MaxPool2d(2)                                                                          # (N, n_channels_out, W/2, H/2)
        elif pool_method == "avg":
            pool = nn.AvgPool2d((2,3))                                                                      # (N, n_channels_out, 1, 1)
        
        conv_block = nn.Sequential(conv, bn, relu, pool)
        #conv_block = nn.Sequential(conv, relu, pool)

        return conv_block
        
        