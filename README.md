# Deep-Climb
A CNN approach to automatically assess bouldering routes difficulty levels

## Installation

### Requirements
To create the environment with all the necessary packages, go at the root of the repository and run:
`conda env create -f environment.yml`

To activate the environment: 
`conda activate deepclimb`

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