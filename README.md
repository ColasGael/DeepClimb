# Deep-Climb
A CNN approach to automatically assess bouldering routes difficulty levels

## Installation

### Requirements
To create the environment with all the necessary packages, go at the root of the repository and run:
`conda env create -f environment.yml`

To activate the environment: 
`conda activate deepclimb`

If you want to use Jupyter notebooks on a remote machine (VM), you need to do this one time set-up.
After you SSH into the VM for the first time, you need to run the following commands in your home directory:

```
cd DeepClimb/
chmod +x ./setup.sh
./setup.sh
```

You will be asked to set up a password for your Jupyter notebook.

Then you will be able to access your VM Jupyter notebooks from your local machine. In your browser, go to: 

`<VM-ip-address>:8888`

### Scrap the data
Folder: "data/raw"

The data has been already scrapped from [moonboard.com](https://www.moonboard.com/Problems/Index). It took around 15 hours.
Scraping data: 26/04/2019.

If you want to update the scraping or scrap a different dataset.
- modify "data/scraper.py" to handle the modifications ;
- download the "selenium" extension corresponding to your browser (for [Chrome](https://sites.google.com/a/chromium.org/chromedriver/downloads)) ;
- create a "moonboard.com" account and create a text file "data/credentials.txt" with your login information ;
- run: `python scraper.py < credentials.txt` in the folder "data".

[Tutorial](https://stanford.edu/~mgorkove/cgi-bin/rpython_tutorials/Scraping_a_Webpage_Rendered_by_Javascript_Using_Python.php) on how to use Selenium to scrap JS-Rendered pages.

Selenium [documentation](https://selenium-python.readthedocs.io/locating-elements.html).

### Preprocess the data

Run: `python setup.py`

It will preprocess and split the scraped data files, for each version of the MoonBoard individually (2016 and 2017).

The script splits the dataset into train/validation/test in the proportion 80%/20%/20%.
The split is random but preserve the class distribution in each split.
The seed used in the split is fixed to give the same split at every run.  

The scrapped data is preprocessed into both the binary (.npy files) and the image (.jpg files) representation of the routes.
The same split is used for both representations.

The split data can be found in "data/binary" and "data/image".

## Train the models

### Launching training

Run: `python train.py --name <model_name>`

You can either use one of the CNN models in "models/CNN_models", or create your custom model and add its name in "train.py" and "test.py".

If you want to change the parameters of the training process, please see: "args.py".

### Track progress on TensorBoard

The training regularly send information to TensorBoard. To start TensorBoard, run the following command from the root directory:
`tensorboard --logdir save --port 5678`

If you are training on your local machine, now open http://localhost:5678/ in your browser. 

If you are training on a remote machine, then run the following command on your local machine:

`ssh -N -f -L localhost:1234:localhost:5678 <user>@<remote> 10`

Here \<user>@\<remote> is the address that you ssh to for your remote machine. 

Then on your local machine, open http://localhost:1234/ in your browser.
    
You should see TensorBoard load with plots of the loss (NLL), Accuracy (Acc), Mean Absolute Error (MAE), and F1-score (F1) for both train and dev sets. 

The dev plots may take some time to appear because they are logged less frequently than the train plots. 
