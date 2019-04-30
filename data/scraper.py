#!/usr/bin/env python3

'''Script to scrap the data from all the problems published on the "moonboard.com" platform

This script was compatible with "moonboard.com" at the following date:
Date: 26/04/2019

Versions of the MoonBoard handled by the scraper:
"MoonBoard 2016" and "MoonBoard 2017" ("40° variant")

Authors:
    Peter Satterthwaite: provided starter code
    Gael Colas
'''

# PACKAGES
# to interact with file and folders
import os 
# to let time for the pages to loads
import time
# progress bar
from tqdm import trange

# scrapping tools
    # to execute JS code (as a browser) from dynamic pages
from selenium import webdriver 
    # to access url without JS scripts faster
import urllib.request # present in standard library
#import json # to read data

# PARAMETERS
version2option = {2016: 'MoonBoard 2016', 2017: 'MoonBoard Masters 2017'} # version of the MoonBoard handled by the scraper
GRADES = ('6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+')


def identification(browser, username, password):
    '''Try to log into 'moonboard.com'
    
    Args:
        'browser' (selenium.webdriver): browser used for the connexion
        'username' (String): 'moonboard.com' username
        'password' (String): 'moonboard.com' username
    
    Remarks:
        You need to create a personal login
    '''
    # find the fields in the page related to the login
    username_field = browser.find_element_by_id("Login_Username") #username form field
    password_field = browser.find_element_by_id("Login_Password") #password form field
    
    # clear the username field
    username_field.clear()
    # fill in your login
    username_field.send_keys(username)
    password_field.send_keys(password)
    
    # submit the login form
    form = browser.find_element_by_id('frmLogin') 
    form.submit()

def choose_MBversion(browser, MBversion):
    '''Navigate the page to select the wanted version of the MoonBoard
    
    Args:
        'browser' (selenium.webdriver): browser used for the connexion
        'MBversion' (int): version (year) of the MoonBoard
    
    Return:
        'n_examples' (int): number of different problems set on this MoonBoard version
        'page_size' (int): number of problems displayed per page
    
    Remarks:
        See the top of this file to see the versions handled by the scraper
    '''
    # check that the version is handled by the scraper
    if MBversion not in version2option:
        raise NotImplementedError("The scrapper does not deal with this version of the MoonBoard.")
    
    # find field related to the version
    Holdsetup = browser.find_element_by_id('Holdsetup')
    
    for option in Holdsetup.find_elements_by_tag_name('option'):
        # find the option corresponding to our version request
        if option.text == version2option.get(MBversion, -1):
            print('Switching to MoonBoard version {}...'.format(MBversion))
            option.click()
            break
    # sleep to let enough time for the page to refresh with the modifications
    time.sleep(3)
    
    # for the 2017 version, we only use the 40° variant
    only40 = True
    if only40 and (MBversion == 2017):
        # find the field related to the board angle
        divConfig = browser.find_element_by_id("divConfig") 

        for option in divConfig.find_elements_by_tag_name('button'):
            # find the angle to deselect
            if option.text == '25° MOONBOARD':
                print('Only select problems on 40° angle board...')
                option.click()
                break
    # sleep to let enough time for the page to refresh with the modifications
    time.sleep(3)
    
    # number of examples matching the request
    n_examples = int(''.join(list(filter(str.isdigit, browser.find_element_by_id('totalProblems').text))))
    # number of examples per page
    index_info = browser.page_source.find("pageSize")
    page_size = int(''.join(list(filter(str.isdigit, browser.page_source[index_info:index_info+13]))))

    return n_examples, page_size  

def get_problem(pb_url, data_file):
    '''Write the data of the current example to the output file
    
    Args:
        'pb_url' (String): url where the example data are stored
        'data_file' (File): file to write the example data to
    
    Remarks:
        As there is no JS script on these pages, we can use urllib for a faster access
    '''
    # indicate if the request reached the given url
    response = None
    # number of times we try to reach the given url
    n_try = 0
    
    # try to reach the guven url
    while response == None and (n_try < 5):
        try:
            # change the header to pretend we are connecting from a browser
            req = urllib.request.Request(pb_url, data=None, headers={'User-Agent': 'Chrome'})
            response = urllib.request.urlopen(req)
            
        except:
            print("Error reading response at url {}, trying again...".format(pb_url))
            n_try += 1
            time.sleep(0.1)
    
    # failure to reach the given url
    if response == None:
        return
    
    # page source code
    html_source = response.read().decode('utf-8')
    
    # begin index of the useful data
    idx = html_source.find('var problem = JSON.parse')
    prob_info = html_source[idx:].split('\n',1)[0]
    # useful data
    temp = prob_info[prob_info.find("'")+1:prob_info.rfind("'")] + '\n'
    
    # replace non-ASCII symbols to avoid issues when writing
    temp = ''.join([i if ord(i) < 128 else 'Z' for i in temp])    
        
    try:
        data_file.write(temp)
        #pb_data = json.loads(temp)
        #print("Current example infos:", pb_data['Name'], '(', pb_data['Grade'], ')', pb_data['UserRating'], 'Stars, by', pb_data['Setter']['Nickname'])

    except Exception as e:
        print(temp)
        raise e

def get_problems(browser, data_file):
    '''Process the examples from the current page
    
    Args:
        'browser' (selenium.webdriver): browser used for the connexion
        'data_file' (File): file to write the examples data to
    
    Return:
        'n_prob' (int): number of processed examples
    '''
    # let the page load
    while True:
        try:
            # list of problems (examples) located on the current page
            probs = browser.find_elements_by_class_name("problem-inner")
            # number of problems on the current page
            n_prob = len(probs)
            
            for i in range(n_prob):
                # get the url of the example page
                ref = probs[i].find_element_by_tag_name('a')
                # process the example
                get_problem(ref.get_attribute('href'), data_file)
            break
            
        except:
            # sleep to let more time for the page to load
            time.sleep(0.1)
            
    return n_prob

def advance_page(browser, page):
    '''Clink on the link to the next page
    
    Args:
        'browser' (selenium.webdriver): browser used for the connexion
        'page' (int): index of the current page
    
    Return:
        'page_next' (int): index of the next page
            page_next = page + 1 or page_next = -1 if the next page is not found
    '''    
    # find the links to other pages
    links = browser.find_elements_by_class_name("k-link")
    
    page_found = False
    
    for i in range(len(links)):
        # check if the current link leads to the next page
        page_found = (links[i].get_attribute('data-page') == str(page + 1))
        if page_found:
            # click on the link to the next page
            links[i].click()
            break
            
    if page_found:      
        return page + 1
    # the next page has not been found
    else:
        return -1 
    
def scrap_moonboard(browser, MBversion, dirName):
    '''Scrap all the examples of the requested version and store the data in the requested directory
    
    Args:
        'browser' (selenium.webdriver): browser used for the connexion
        'MBversion' (int): version (year) of the MoonBoard
        'dirName' (String): path to the directory where we want the data to be stored
    
    Remarks:
        The data is stored as a text file in: dirName/'<MBversion>_moonboard_data.txt'
        The examples are stored as lines
    '''    
    
    # select the correct version
    n_examples, page_size = choose_MBversion(browser, MBversion)
    
    print("{} examples detected for MoonBoard version {}.".format(n_examples, MBversion))
    
    # path to the output file
    path_out = os.path.join(dirName, "{}_moonboard_data.txt".format(MBversion))
    
    # we do not overwrite this file: change the following line if you want to
    overwrite = False
    # check if there is already a version of this file
    if os.path.exists(path_out):
        if overwrite:
            print("Output file for version {} already exists, removing...".format(MBversion))
            os.remove(path_out)
        else:
            print("Output file for version {} already exists, exiting...".format(MBversion))
            return
            #raise FileExistsError("Output file for version {} already exists.".format(MBversion))
    
    # create the output file
    with open(path_out, 'a') as data_file:
        print("Writing to {}...".format(path_out))

        # max number of pages to scroll
        n_pages = int(n_examples/ page_size) + 1
        
        # current page number
        page = 1
        n_examples_processed = 0
        
        # progress bar showing the number of pages processed
        pbar = trange(n_pages)
        for i in pbar:
            # process the current page's examples
            n_prob = get_problems(browser, data_file)
            # go to the next page
            page = advance_page(browser, page)
            # sleep to let the next page load
            #time.sleep(1)
            
            # update the number of processed examples
            n_examples_processed += n_prob
            
            # last page reached
            if page < 0:
                print('Last page reached.')
                break
                
        print('{} / {} examples processed'.format(n_examples_processed, n_examples))
                
def main(dirName):    
    try:
        # create a directory to store the scraping outputs
        os.mkdir(dirName)
        print("Directory '{}' created.".format(dirName))
    except FileExistsError:
        print("Directory '{}' already exists.".format(dirName))

    invalidCredentials = True
    try:
        # Chrome browser used to run the JS pages
        browser = webdriver.Chrome()
        # connect to the login page
        url_ident = 'https://www.moonboard.com/Account/Login'
        browser.get(url_ident)
        
        while invalidCredentials:
            # ask the user for its login
            username = input("Username:")
            password = input("Password:")
            
            # try to login
            identification(browser, username, password)

            url_probs = 'https://www.moonboard.com/Problems/Index'
            browser.get(url_probs)
            
            # check that we reached the desired page ie that we logged in
            invalidCredentials = (browser.current_url != url_probs)
            
            if invalidCredentials:
                print(browser.current_url)
                print("Invalid Credentials: please try again.")
            else:
                print("\nLogged in!")
        # sleep to let the page load
        time.sleep(5)
        
        # print source code of dynamic page
        #print(browser.page_source)
        
        # version of the MoonBoard handled by the scraper
        MBversions = version2option.keys()        
                
        for MBversion in MBversions:
            # scrap all the examples of the current version
            scrap_moonboard(browser, MBversion, dirName)
        
        # clean up: close the browser
        browser.close()
    
    except Exception as e:
        # clean up: close the browser
        browser.close()
        raise e
        
if __name__ == "__main__":
    # directory where we want to store the scraped data
    dirName = 'raw' 
    main(dirName)