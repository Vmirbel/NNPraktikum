#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.sine import Sine
from model.mlp_sine import MultilayerPerceptronSine

import numpy as np
import matplotlib.pyplot as plt


def main():
    data = Sine()
    myMLPerceptron = MultilayerPerceptronSine(data.trainingSet,
                                              data.validationSet,
                                              data.testSet,
                                              hiddenLayers=[8],
                                              outputDim=1,
                                              epochs=150)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMultilayer perceptron has been training..")
    myMLPerceptron.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPerceptron.evaluate()

    # Report the result
    print("=========================")

    x = np.arange(-np.pi, np.pi, np.pi/100)
    y = np.sin(x)
    plt.plot(x, y)  # the reference sine curve
    plt.plot(data.testSet.input[:, 0],  mlpPred, 'r+')  # MLP approximation
    plt.show()

if __name__ == '__main__':
    main()
