#!/usr/bin/env python3

'''Script to preprocess the scraped data into a clean dataset
    input = binary matrix representing a problem on the moonboard
    label = class number representing the grade of the problem

This script was compatible with (the String representation of a problem information on) "moonboard.com" at the following date:
Date: 04/2019

Grades handled by the preprocessing:
('6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+')

Authors:
    Gael Colas
'''

# PACKAGES
# to interact with file and folders
import os 
# to parse the strings
import re
# to create the matrix representations of examples
import numpy as np
# to save data files
import ujson as json

# PARAMETERS
# list of all the fields describing the problem in the order they appear in the raw scraped file
FIELDS = ["Method:", "Name:", "Grade:", "UserGrade:", "MoonBoardConfiguration:", "MoonBoardConfigurationId:", "Setter:", "FirstAscender:", "Rating:", "UserRating:", 
"Repeats:", "Attempts:", "Holdsetup:", "IsBenchmark:", "IsAssessmentProblem:", "ProblemType:", "Moves:", "Holdsets:", "Locations:", "RepeatText", "NumberOfTries::",
"NameForUrl:", "Upgraded:", "Downgraded:", "Id:", "ApiId:", "DateInserted:", "DateUpdated:", "DateDeleted:", "DateTimeString:"]
# fields we want to extract
USEFUL_FIELDS = ["Grade:", "UserGrade:", "Moves:"]
# MoonBoard grid properties
GRID_DIMS = (18, 11) # dimensions
COLUMNS = [str(chr(ascii_nb)) for ascii_nb in range(ord('A'), ord('K')+1)]
ROWS = [str(i) for i in range(GRID_DIMS[0],0,-1)]

def moves2binary(moves_list, moves_type, move2coord):
    '''Convert a list of moves into a binary representation of the problem
    
    Args:
        'moves_list' (list of String): list of the moves of the problem
        'moves_type' (list of String): list of the type of each move (start: 0, intermediate: 1 or end: 2 move)
            moves_type[2*k] = "true" if moves_list[k] is a start move, "false" otherwise
            moves_type[2*k+1] = "true" if moves_list[k] is a end move, "false" otherwise
        'move2coord' (dict: String -> (int,int)): dictionary mapping the move String to its coordinate in the binary grid
        
    Return:
        'x_binary' (np.array, shape=GRID_DIMS, dtype=int): binary matrix representing the problem on the MoonBoard
            x_binary[i,j] = 1 if you can use this move in the problem, 0 otherwise
        'x_type' (np.array, shape=GRID_DIMS, dtype=int): int matrix representing the type of each move
            x_type[i,j] = 0 if this is a start move
            x_type[i,j] = 1 if this is an intermediate move
            x_type[i,j] = 2 if this is an end move
            x_type[i,j] = -1 if this is not a move of the problem
    '''
    assert(2*len(moves_list) == len(moves_type))

    # empty binary grid
    x_binary = np.zeros((GRID_DIMS[0], GRID_DIMS[1]), dtype=int)
    # empty type of move grid
    x_type = -np.ones((GRID_DIMS[0], GRID_DIMS[1]), dtype=int)
    
    for k, move in enumerate(moves_list):
        # grid coordinates of the move
        i, j = move2coord[move]
        # add the corresponding available hold to the binary grid
        x_binary[i,j] = 1
        
        # indicate the type of move
        if moves_type[2*k] == "true": # start move
            x_type[i,j] = 0
        elif moves_type[2*k+1] == "true": # end move
            x_type[i,j] = 2
        else: # intermediate move
            x_type[i,j] = 1
    
    return x_binary, x_type

def read_problem(problem_string, field2pos, move2coord, grade2class):
    '''Convert a String representation of the problem information into a (input, label)-pair
    
    Args:
        'problem_string' (String): String representation of the problem
        'field2pos' (dict: String -> int): dictionary mapping information fields to their order of appearance in the String representation
        'move2coord' (dict: String -> (int,int)): dictionary mapping the move String to its coordinate in the binary grid
        'grade2class' (dict: String -> int): dictionary mapping the grade String to its corresponding int class label
    
    Return:
        'x_binary' (np.array, shape=GRID_DIMS, dtype=int): binary matrix representing the problem on the MoonBoard
            x_binary[i,j] = 1 if you can use this move in the problem, 0 otherwise
        'x_type' (np.array, shape=GRID_DIMS, dtype=int): int matrix representing the type of each move
            x_type[i,j] = 0 if this is a start move
            x_type[i,j] = 1 if this is an intermediate move
            x_type[i,j] = 2 if this is an end move
            x_type[i,j] = -1 if this is not a move of the problem
        'y' (int): class label of the problem given by the setter
            y = None, if the grade of the problem is not handled by the preprocessing
        'y_user' (int): class label of the problem given by the user
            y_user = -1, if no grade has been given to the problem by the user
    '''
    # clean the String
    problem_string = problem_string.replace('"', '')
        
    # extract the useful fields' information
    for field in USEFUL_FIELDS:
        # position of the field in the list of FIELDS
        field_ind = field2pos[field] 
        # where the useful information starts in the string
        field_start = problem_string.find(FIELDS[field_ind]) + len(FIELDS[field_ind])
        # where the useful information ends in the string
        field_end = problem_string.find(FIELDS[field_ind+1], field_start) -1 
        # information stored in the field
        field_info = problem_string[field_start:field_end]
                
        if field == "Moves:":
            # find the list of moves 
            moves_list = re.findall("[A-Z]\d+", field_info)
            # type of move (start, intermediate or end)
            moves_type = re.findall("true|false", field_info)            
            # convert to binary matrix
            x_binary, x_type = moves2binary(moves_list, moves_type, move2coord)
        
        elif field == "Grade:":
            # convert to a class label
            y = grade2class.get(field_info, None)
        
        elif field == "UserGrade:":
            # convert to a class label
            y_user = grade2class.get(field_info, -1)        
    
    return x_binary, x_type, y, y_user
    
def read_problems(raw_data_path, GRADES): 
    '''Convert the scraped data into a clean dataset
    
    Args:
        'raw_data_path' (String): path to the scraped data
    
    Return:
        'X_binary' (np.array, shape=(n_examples, prod(GRID_DIMS)), dtype=int): design matrix
            X_binary[k, :] = flatten binary matrix representing the problem k on the MoonBoard
        'X_type' (np.array, shape=(n_examples, prod(GRID_DIMS)), dtype=int): int matrix storing the type of each move for every problem
            X_type[k, :] = flatten int matrix representing the type of each move for problem k
        'Y' (np.array, shape=(n_examples,), dtype=int): label vector
            Y[k] = class label of problem k given by the setter
        'Y_user' (np.array, shape=(n_examples,), dtype=int): label given by the user vector
            Y_user[k] = class label of problem k given by the user
        'grade2class' (dict: String -> int): dictionary mapping the grade String to its corresponding int class label
    '''
    # map the useful fields to their position in the list of fields
    field2pos = {field: FIELDS.index(field) for field in USEFUL_FIELDS}
    
    # map move to its coordinates in the grid
    move2coord = {col+row: (i,j) for i, row in enumerate(ROWS) for j, col in enumerate(COLUMNS)}

    # map grade to the corresponding class
    grade2class = {grade:i for i, grade in enumerate(GRADES)}
    
    # read the text file
    data = open(raw_data_path, 'r')
    
    problems = data.readlines()
    
    # number of examples
    n_examples = len(problems)
    
    # dataset: 1 row = 1 example
    X = np.zeros((n_examples, np.prod(GRID_DIMS)), dtype=int)
    # type of move: 1 row = 1 example
    X_type = np.zeros((n_examples, np.prod(GRID_DIMS)), dtype=int)
    # labels: 1 label = 1 setter grade
    Y = np.zeros((n_examples,), dtype=int)
    # user predicted grade: 1 label = 1 user grade
    Y_user = np.zeros((n_examples,), dtype=int)
    
    i = 0
    # read one line (corresponding to one problem information) at a time
    for problem_string in problems:       
        # preprocess the corresponding problem
        x_binary, x_type, y, y_user = read_problem(problem_string, field2pos, move2coord, grade2class)
        
        if y != None:
            # store the example
            X[i,:] = np.reshape(x_binary, (-1,))
            X_type[i,:] = np.reshape(x_type, (-1,))
            Y[i] = y
            Y_user[i] = y_user
            
            i += 1
        
    # cleanup: close the file
    data.close()
    
    return X[:i,:], X_type[:i,:], Y[:i], Y_user[:i], grade2class
    
def main(rawDirName, ppDirName, filenames_out, VERSIONS, GRADES):
    print("\n BINARY PREPROCESSING\n")

    try:
        # create a directory to store the preprocessed data
        os.mkdir(ppDirName)
        print("Directory '{}' created.".format(ppDirName))
    except FileExistsError:
        print("Directory '{}' already exists.".format(ppDirName))

    for MBversion in VERSIONS:
        print("{:-^100}".format("---Preprocessing for MoonBoard version {}---".format(MBversion)))
        # path to raw datafile
        path_in = os.path.join(rawDirName, "{}_moonboard_data.txt".format(MBversion))
        
        # path to preprocessed files
        ppVersionDirName = os.path.join(ppDirName, MBversion)
        try:
            # create a directory to store the preprocessed data
            os.mkdir(ppVersionDirName)
            print("Directory '{}' created.".format(ppVersionDirName))
        except FileExistsError:
            print("Directory '{}' already exists.".format(ppVersionDirName))
        
        print("Preprocessing the scraped data...")
        # preprocess the raw data files
        X, X_type, y, y_user, grade2class = read_problems(path_in, GRADES)
        
        # number of examples 
        n_examples = X.shape[0]
        print("There are {} examples (distinct MoonBoard version {} problems).".format(n_examples, MBversion))
            
        print("Writing the preprocessed data to disk...")
        filenames_out = filenames_out + ["grade2class"]
        # save the preprocessed data
        for k, data in enumerate([X, X_type, y, y_user, grade2class]):
            # path to the output file
            path_out = os.path.join(ppVersionDirName, filenames_out[k])
            
            if isinstance(data,(np.ndarray)):
                # save the numpy arrays
                np.save(path_out, data)
            else:
                with open(path_out+'.json', 'w') as json_file:  
                    # save the dictionaries
                    json.dump(data, json_file)
            print(filenames_out[k], "saved!")
        
if __name__ == "__main__":
    # versions of the MoonBoard handled
    VERSIONS = ["2016", "2017"]
    # problems' grades considered
    GRADES = ('6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+')
    # directory where the scraped files are stored
    rawDirName = 'raw' 
    # directory where to store the preprocessed files
    ppDirName = 'binary'
    # filenames for the preprocessed data
    filenames = ["X", "X_type", "y", "y_user"] 
    
    main(rawDirName, ppDirName, filenames, VERSIONS, GRADES)     