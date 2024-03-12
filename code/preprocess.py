import numpy as np
from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()
   
    train_inputs = train_inputs.reshape(len(train_inputs), -1)
    test_inputs = test_inputs.reshape(len(test_inputs), -1)
    train_inputs = train_inputs / 255.0
    test_inputs = test_inputs / 255.0
    ...
    
    train_inputs = Tensor(train_inputs)
    train_labels = Tensor(train_labels)
    test_inputs = Tensor(test_inputs)
    test_labels = Tensor(test_labels)
    ...
    return train_inputs, train_labels, test_inputs, test_labels