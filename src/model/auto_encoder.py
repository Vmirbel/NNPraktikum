# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class AutoEncoder:
    """
    Abstract class of a classifier
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, trainingSet, validationSet):
        # Train procedures of the AutoEncoder
        pass
