#!/usr/bin/env python3

# PACKAGES
# to interact with file and folders
import os 
# to parse the strings
import re
# to create the matrix representations of examples
import numpy as np

# PARAMETERS
# versions of the MoonBoard handled
VERSIONS = ["2016", "2017"]
# list of all the fields describing the problem in the order they appear in the raw scraped file
FIELDS = ["Method:", "Name:", "Grade:", "UserGrade:", "MoonBoardConfiguration:", "MoonBoardConfigurationId:", "Setter:", "FirstAscender:", "Rating:", "UserRating:", 
"Repeats:", "Attempts:", "Holdsetup:", "IsBenchmark:", "IsAssessmentProblem:", "ProblemType:", "Moves:", "Holdsets:", "Locations:", "RepeatText", "NumberOfTries::",
"NameForUrl:", "Upgraded:", "Downgraded:", "Id:", "ApiId:", "DateInserted:", "DateUpdated:", "DateDeleted:", "DateTimeString:"]
# fields we want to extract
USEFUL_FIELDS = ["Grade:", "UserGrade:", "Moves:"]
# problems' grades considered
GRADES = ('6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+')
# MoonBoard grid properties
GRID_DIMS = (18, 11) # dimensions
COLUMNS = [str(chr(ascii_nb)) for ascii_nb in range(ord('A'), ord('K')+1)]
ROWS = [str(i) for i in range(GRID_DIMS[0],0,-1)]

def moves2binary(moves_list, moves_type, move2coord):
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
    # clean the String
    problem_string = problem_string.replace('"', '')
        
    # extract the useful fields' information
    for field in USEFUL_FIELDS:
        # position of the field in the list of FIELDS
        field_ind = field2pos[field] 
        # where the useful information starts in the string
        field_start = problem_string.find(FIELDS[field_ind]) + len(FIELDS[field_ind]) +1 
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
    
def read_problems(raw_data_path): 
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
    
    # read one line (corresponding to one problem information) at a time
    for i, problem_string in enumerate(problems):       
        # preprocess the corresponding problem
        x_binary, x_type, y, y_user = read_problem(problem_string, field2pos, move2coord, grade2class)
        
        if y != None:
            # store the example
            X[i,:] = np.reshape(x_binary, (-1,))
            X_type[i,:] = np.reshape(x_type, (-1,))
            Y[i] = y
            Y_user[i] = y_user
        
    # cleanup: close the file
    data.close()
    
    return X, X_type, Y, Y_user
    
def main(rawDirName, ppDirName):

    try:
        # create a directory to store the preprocessed data
        os.mkdir(ppDirName)
        print("Directory '{}' created.".format(ppDirName))
    except FileExistsError:
        print("Directory '{}' already exists.".format(ppDirName))

    for MBversion in VERSIONS:
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
        X, X_type, y, y_pred = read_problems(path_in)
        
        print("Writing the preprocessed data to disk...")
        filenames_out = ["X", "X_type", "y", "y_pred"]
        # save the preprocessed data
        for i, array_out in enumerate([X, X_type, y, y_pred]):
            # path to the output file
            path_out = os.path.join(ppVersionDirName, filenames_out[i]+".csv")
            np.savetxt(path_out, array_out, delimiter=",")
    
if __name__ == "__main__":
    # directory where the scraped files are stored
    rawDirName = 'raw' 
    # directory where to store the preprocessed files
    ppDirName = 'binary'
    
    main(rawDirName, ppDirName)     