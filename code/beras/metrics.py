import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    """
    TODO:
        - call
    """
    def call(self, probs, labels):
        ## Compute and return the categorical accuracy of your model 
        
        probs = np.argmax(probs, axis=1) # get the index of the max probability
        labels = np.argmax(labels, axis=1) # get the index of the one-hot encoded vectors
        return np.mean(probs == labels)

