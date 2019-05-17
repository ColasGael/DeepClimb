#!/usr/bin/env python3

"""Convert the binary representation into an image representation of the examples
    input = binary matrix indicating the available holds for a problem
    output = image where the available holds are circled on the MoonBoard template image

Original size of images: (C, H, W) = (3, 1000, 650)
    
Color code:
    - start holds: green
    - intermediate holds: blue
    - end holds: red
    
Authors:
    Gael Colas
"""

# PACKAGES
# to interact with file and folders
import os 
# to handle the matrix representations of examples
import numpy as np
# progress bar
from tqdm import trange
# to handle images
from PIL import Image, ImageDraw

# MoonBoard grid properties
GRID_DIMS = (18, 11) # dimensions

def draw_circle_wrapper(im, center, radius, width, color=(255,0,0,255), draw=None):
    '''Draw a circle on the image, the image is modified in place
    
    Args:
        'im' (PIL.Image): image to draw on
        'center' (tuple of int, size=2): (x, y) pixel coordinates of the circle's center
        'radius' (int): radius (in pixels) of the circle
        'width' (int): width of the circle
        'color' (tuple of int, size=4, default=(255,0,0,255)): RGB-color of the circle (default is 'red')
        'draw' (ImageDraw.Draw): interface used to draw
    '''
    # create a new drawing interface is None is provided
    if draw is None:
        draw = ImageDraw.Draw(im)
    
    # circle parameters
    x, y = center
    r = radius
    
    # add some width to the circle by drawing multiple bigger circles
    for i in range(width):
        draw.ellipse((x-r-i, y-r-i, x+r+i, y+r+i), outline=color)

def resize_wrapper(im, new_width=256, new_height=None):
    '''Resize the input image while preserving the aspect ratio
    
    Args:
        'im' (PIL.Image): image to resize
        'new_width' (int, default=256): width of the resized image
        'new_height' (int, default=None): height of the resized image
    
    Return:
        'im_resized' (PIL.Image): resized image with width 'new_width'  

    Remark:
        The resize operation preserve the aspect ratio.
        If 'new_height' is not None, we use this argument to resize the image.
        Otherwise, we use 'new_width' to resize the image.
    '''
    # original image dimensions
    width, height = im.size

    # resized image dimensions: preserve the aspect ratio
    if new_height is not None:
        new_height = int(new_height)
        new_width = int(new_height* width / height)
    else:
        new_width = int(new_width)
        new_height = int(new_width* height / width)
    
    # resized image
    im_resized = im.resize((new_width, new_height), Image.BILINEAR)
    
    return im_resized
            
def binary2image(x_binary, x_type, template_im):
    '''Create a image representing the problem with available holds circled
        as described by the binary representation
        
    Args:
        'x_binary' (np.array, shape=GRID_DIMS, dtype=int): binary matrix representing the problem on the MoonBoard
            x_binary[i,j] = 1 if you can use this move in the problem, 0 otherwise
        'x_type' (np.array, shape=GRID_DIMS, dtype=int): int matrix representing the type of each move
            x_type[i,j] = 0 if this is a start move
            x_type[i,j] = 1 if this is an intermediate move
            x_type[i,j] = 2 if this is an end move
            x_type[i,j] = -1 if this is not a move of the problem
        'template_im' (PIL.Image): blank (no holds circled) template of the MoonBoard wall
        
    Return:
        'x_im' (PIL.Image): image representation of the problem with available holds circled
    
    Remark:
        Color code: start holds = green ; intermediate holds = blue ; end holds = red
    '''
    # copy the image: not to overwrite it
    x_im = template_im.copy()
    # image dimension
    width, height = x_im.size
    
    # HYPERPARAMETERS: tuned on the MoonBoard image
    offset_x = 2*int(width/14) # x-position of first hold
    offset_y = 2*int(height/23) # y-position of first hold
    
    dx = int(width/12.8) # x-distance between two holds
    dy = int(height/19.67) # y-distance between two holds
    
    radius = min(dx, dy)//2 # radius of circle
    w = max(radius//5, 1) # width of circle
    
    # create a drawing interface
    draw = ImageDraw.Draw(x_im)

    # extract position of available holds
    i_array, j_array = np.nonzero(x_binary)

    # loop over the holds
    for k in range(i_array.size):
        # hold's grid coordinates
        i, j = i_array[k], j_array[k]
        # corresponding position on the image
        center = (j*dx + offset_x, i*dy + offset_y)
        
        # choose the color of the circle corresponding to the type of the hold
        if x_type[i,j] == 0: # start hold
            color = (0, 255, 0, 255) # green
        elif x_type[i,j] == 1: # intermediate hold
            color = (0, 0, 255, 255) # blue
        elif x_type[i,j] == 2: # end hold
            color = (255, 0, 0, 255) # red
        else:
            raise ValueError("Inconsistency between type of holds and holds' locations. \nAre they from the same example?")
        
        # draw a circle of the right color centered on the hold
        draw_circle_wrapper(x_im, center, radius, w, color, draw)
    
    # deleter the drawing interface
    del draw
    
    return x_im
    
def main(rawDirName, binDirName, imDirName, splits, VERSIONS):
    print("\n IMAGE PREPROCESSING\n")

    try:
        # create a directory to store the preprocessed image data
        os.mkdir(imDirName)
        print("Directory '{}' created.".format(imDirName))
    except FileExistsError:
        print("Directory '{}' already exists.".format(imDirName))

    for MBversion in VERSIONS:
        MBversion = str(MBversion)
        print("{:-^100}".format("---Image Preprocessing for MoonBoard version {}---".format(MBversion)))

        # load the blank template of the MoonBoard wall
        template_im = Image.open(os.path.join(rawDirName, "{}_moonboard_empty.png".format(MBversion)))
        
        # convert from RGBA (.png) to RGB (.jpg)
        template_im = template_im.convert('RGB')

        # path to the datafiles
        path_in = os.path.join(binDirName, MBversion)
        
        # path to preprocessed images
        path_out = os.path.join(imDirName, MBversion)
        try:
            # create a directory to store the preprocessed images
            os.mkdir(path_out)
            print("Directory '{}' created.".format(path_out))
        except FileExistsError:
            print("Directory '{}' already exists.".format(path_out))
            
        for split_name in splits:
            # load the binary data
            X = np.load(os.path.join(path_in, "X_{}.npy".format(split_name)))
            X_type = np.load(os.path.join(path_in, "X_type_{}.npy".format(split_name)))
            y = np.load(os.path.join(path_in, "y_{}.npy".format(split_name)))
            
            # path to preprocessed images per split
            splitDirName = os.path.join(path_out, split_name)
            try:
                # create a directory to store the preprocessed images per split
                os.mkdir(splitDirName)
                print("Directory '{}' created.".format(splitDirName))
            except FileExistsError:
                print("Directory '{}' already exists.".format(splitDirName))
            
            print("Preprocessing the binary {} data into artificial images...".format(split_name))
            # loop over the examples
            for k in trange(X.shape[0]):
                # binary matrix representation of the example
                x_binary = np.reshape(X[k,:], GRID_DIMS)
                x_type = np.reshape(X_type[k,:], GRID_DIMS)
                
                # build the image representation of the example
                x_im = binary2image(x_binary, x_type, template_im)
                
                # save the image: "<label>_<split>_<example_nb>.png"
                im_name = "{}_{}_{}.jpg".format(y[k], split_name, k)
                
                # save the image
                x_im.save(os.path.join(splitDirName, im_name), "JPEG")
            
if __name__ == "__main__":
    # versions of the MoonBoard handled
    VERSIONS = ["2016", "2017"]
    # directory where the blank template images of the MoonBoard walls are stored
    rawDirName = 'raw'
    # directory where the binary matrices are stored
    binDirName = 'binary' 
    # directory where to store the preprocessed images
    imDirName = 'image'
    # splits to preprocess
    splits = ("train", "val", "test")
    
    main(rawDirName, binDirName, imDirName, splits, VERSIONS)
            