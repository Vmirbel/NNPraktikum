# -*- coding: utf-8 -*-

from random import random
from model.classifier import Classifier

__author__ = "Vladimir Belyaev"  # Adjust this when you copy the file
__email__ = "vladimir.belyaev@student.kit.edu"  # Adjust this when you copy the file


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.

    The method is that it will randomly decide the digit is a "7" or not
    based on the probability 'byChance'.
    """

    def __init__(self, train, valid, test, byChance=0.5):

        self.byChance = byChance

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

    def train(self):
        # Do nothing
        pass

    def classify(self, testInstance):
        # byChance is the probability of being correctly recognized
        return random() < self.byChance

    def evaluate(self):
        return list(map(self.classify, self.testSet.input))
