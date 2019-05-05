# to interact with file and folders
import os 
# to handle the matrix representations of examples
import numpy as np
# to save the model parameters
import ujson as json

class NaiveBayes:
    """Naive Bayes multi-variate Bernouilli event model with Laplace smoothing
    
    Attributes:
        'parameters' (dict): dictionary storing the model parameteres
        'weights' (dict, default={}): how much to weight each class 
            weight[c] = weight of class c

    Remarks:
        parameters['log_phi'] (np.array, shape=(C,)): log(p(y = c))
        parameters['log_phi_j_1'] (np.array, shape=(C, d)): log(p(x_j = 1 | y = c))
        parameters['log_phi_j_0'] (np.array, shape=(C, d)): log(p(x_j = 0 | y = c))
        parameters['classes'] (list of int, len <= C): classes present in the train-set
    """
    
    def __init__(self, weights={}):
        self.parameters = {}
        self.weights = weights
    
    def fit(self, X_train, y_train):
        """Fit the Naive Bayes classifier on the train-set and update 'self.parameters'
        
        Args:
            'X_train' (np.array, shape=(n_train, d): train design matrix
            'y_train' (np.array, shape=(n_train,): label vector
            
        Remarks:
            To avoid underflow errors, we store the log-probabilities
            Assumption: all the classes are represented in 'y_train'
        """    
        # number of training examples and input size
        n_train, d = X_train.shape
        # number of classes
        C = np.max(y_train) +1

        # initialization of the model parameters
        self.parameters['log_phi'] = np.zeros(C,)
        self.parameters['log_phi_j_1'] = np.zeros((C, d))
        self.parameters['log_phi_j_0'] = np.zeros((C, d))
        self.parameters['classes'] = [int(c) for c in set(y_train)]
        
        # loop over all the classes
        for c in self.parameters['classes']:
            mask = (y_train == c)
            # p(y=c)
            phi = np.sum(mask) * self.weights.get(c, 1)
            self.parameters['log_phi'][c] = phi
            # p(x_j = 1|y=c)
            phi_j = (np.sum(X_train[mask, :], axis = 0) + 1)/(mask.shape[0] + 2)
            self.parameters['log_phi_j_1'][c,:] = np.log(phi_j)
            self.parameters['log_phi_j_0'][c,:] = np.log(1-phi_j)
        
        # normalization
        mask = [(c in self.parameters['classes']) for c in range(C)]
        self.parameters['log_phi'][mask] = np.log(self.parameters['log_phi'][mask] / np.sum(self.parameters['log_phi']))
        
    def predict(self, X):
        """Predict the label of each example using the model parameters
        
        Args:
            'X' (np.array, shape=(n, d): design matrix
            
        Return:
            'y_pred' (np.array, shape=(n,): predicted labels
        """
        # check that the model has been trained
        assert self.parameters, "The model has not been trained: run 'model.fit(X_train, y_train)' or load pre-trained parameters 'model.load(model_save_path)'"
        
        # most likely classes
        y_pred = None
        # corresponding unormalized log-probabilities
        best_prob = None
        
        # loop over all the classes
        for c in self.parameters['classes']:
            prob_c = np.sum(X * self.parameters['log_phi_j_1'][c,:] + (1-X) * self.parameters['log_phi_j_0'][c,:], axis=1) + self.parameters['log_phi'][c]
            
            if (y_pred is None):
                y_pred = c*np.ones(X.shape[0])
                best_prob = prob_c
                
            # update the most likely class
            mask = best_prob < prob_c
            y_pred[mask] = c
            best_prob[mask] = prob_c[mask]
            
        return y_pred
        
    def save(self, save_name="NBmodel.json", model_save_path="./save/NB"):
        """Save the current model parameters
        
        Args:
            'save_name' (String, default="NBmodel.json"): filename of the JSON file storing the parameters
            'model_save_path' (String, default="../save"): where to save the model parameters
        """
        try:
            # create a directory to store the trained parameters
            os.mkdir(model_save_path)
        except FileExistsError:
            pass
        
        with open(os.path.join(model_save_path, save_name), 'w') as json_file:  
            json.dump(self.parameters, json_file)
            
    def load(self, save_name="NBmodel.json", model_save_path="./save/NB"):
        """Load the pre-trained model parameters
        
        Args:
            'save_name' (String, default="NBmodel.json"): filename of the JSON file storing the parameters
            'model_save_path' (String, default="../save"): where the model parameters are saved
        """
        with open(os.path.join(model_save_path, save_name), 'r') as json_file:  
            self.parameters = json.load(json_file)
        
        # convert from list to np.array
        for key, parameter in self.parameters.items():
            if key != 'classes':
                self.parameters[key] = np.array(parameter)
            
if __name__ == "__main__":
    # data path
    data_path = "./data/binary/2016"
    #current_path = os.path.realpath(__file__)
    # import the train data
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    # import the validation data
    X_val = np.load(os.path.join(data_path, "X_val.npy"))
    y_val = np.load(os.path.join(data_path, "y_val.npy"))
    
    # new model
    model = NaiveBayes()
    
    # train the model
    model.fit(X_train, y_train)
    
    # save the model
    model.save(model_save_path="./save/NB")
    del model
    
    # new model
    model_new = NaiveBayes()
    model_new.load(model_save_path="./save/NB")
    
    # predict
    y_train_pred = model_new.predict(X_train)
    y_val_pred = model_new.predict(X_val)
    
    # measure the accuracy
    acc_train = np.sum(y_train == y_train_pred)/y_train.shape[0]
    acc_val = np.sum(y_val == y_val_pred)/y_val.shape[0]
    
    print("The Naive Bayes model achieves {:.2f}% training accuracy.".format(acc_train*100))
    print("The Naive Bayes model achieves {:.2f}% validation accuracy.".format(acc_val*100))