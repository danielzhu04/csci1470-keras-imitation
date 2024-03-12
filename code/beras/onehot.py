import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    TODO:
        - fit
        - call
        - inverse

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        ## the unique labels as keys and their one hot encodings as values

        data = np.unique(data)
        size = len(data)
        self.label_to_ohe = {}
        self.ohe_to_label = {}
        for i, label in enumerate(data):
            ohe = np.zeros(size)
            ohe[i] = 1
            self.label_to_ohe[label] = ohe
            self.ohe_to_label[tuple(ohe)] = label

    def call(self, data):
        encoded = np.array([self.label_to_ohe[label] for label in data])
        return encoded

    def inverse(self, data):
        decoded = np.array([self.ohe_to_label[tuple(ohe)] for ohe in data])
        return decoded