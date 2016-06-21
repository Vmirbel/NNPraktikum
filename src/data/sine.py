# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet


class Sine(object):
    """
    Small subset (5000 instances) of synthesis data to reproduce sine()

    Parameters
    ----------
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, numTrain=3000, numValid=1000, numTest=1000):

        # Generate data
        fr = 200  # sample rate
        sf = 2  # the signal frequency

        trainingData = []
        while len(trainingData) < (numTrain + numValid + numTest):
            for i in xrange(fr):
                y = np.sin(2 * np.pi * sf * (i/fr))
                trainingData.append([np.sin(y), y])

        print(len(trainingData))
        shuffle(trainingData)

        # Assign data to sets
        self.trainingSet = DataSet(np.array(trainingData[:numTrain]),
                                   one_hot=False)
        self.validationSet = DataSet(np.array(trainingData[numTrain:-numTest]),
                                     one_hot=False)
        self.testSet = DataSet(np.array(trainingData[-numTest:]),
                               one_hot=False)
