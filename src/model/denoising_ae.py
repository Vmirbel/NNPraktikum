# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, train, valid, test, learning_rate=0.1, epochs=30):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        pass

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        pass

    def _get_weights(self):
        """
        Get the weights (after training)
        """

        pass
