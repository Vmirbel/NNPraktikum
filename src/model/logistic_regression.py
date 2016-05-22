# -*- coding: utf-8 -*-

__author__ = "Vladimir Belyaev"  # Adjust this when you copy the file
__email__ = "vladimir.belyaev@student.kit.edu"  # Adjust this when you copy the file

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.layer = LogisticLayer(len(self.trainingSet.input[1]), 1,  is_classifier_layer=True, learningRate=learningRate)
        # create an array to save performance after each epoch
        self.perf = []

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement training method "epochs" times
        # Please using LogisticLayer class

        for epochIdx in range(self.epochs):
            for xi, target in zip(self.trainingSet.input, self.trainingSet.label):
                self.classify(xi)
                self.layer.computeDerivative(target, self.layer.weights)
                self.layer.updateWeights()

            lrPred = self.evaluate()
            self.perf.append(accuracy_score(self.testSet.label, lrPred)*100)

            if verbose:
                print("Epoch: %d" % epochIdx)

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

        # Here you have to implement classification method given an instance
        self.layer.forward(testInstance)
        if self.layer.outp > 0.5:
            return 1
        else:
            return 0

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
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
