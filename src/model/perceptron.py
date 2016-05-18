# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score

from util.activation_functions import Activation
from model.classifier import Classifier

__author__ = "Vladimir Belyaev"  # Adjust this when you copy the file
__email__ = "vladimir.belyaev@student.kit.edu"  # Adjust this when you copy the file

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : ndarray
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

        self.errors_ = []
        # add bias values ("1"s) at the beginning of all data sets
        np.insert(self.trainingSet.input, 0, 1, axis=1)
        np.insert(self.validationSet.input, 0, 1, axis=1)
        np.insert(self.testSet.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """
        Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy*100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        pass
        for epochIdx in range(self.epochs):
            errors = 0
                error = self.learningRate * (target - self.fire(xi))
                errors += int(error != 0.0)
            # online learning: updating weights after seeing 1 instance
            if verbose:
                print("Epoch: %d --> Errors: %d" % (epochIdx, errors))
        return self

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise
        pass
        return self.fire(testInstance)


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def _fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight) + self.w0)
