#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from data.mnist_seven import MNISTSeven
# from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
# from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator



def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    # myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                       data.validationSet,
    #                                       data.testSet)
    # Uncomment this to make your Perceptron evaluated
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                         data.validationSet,
                                         data.testSet,
                                         learningRate=0.005, # default learningRate: 0.005
                                         epochs=100)         # default epochs: 30

    # Train the classifiers
    print("=========================")
    print("Training..")

    # print("\nStupid Classifier has been training..")
    # myStupidClassifier.train()
    # print("Done..")

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")

    # plot a graph with a number of errors after each epoch
    plt.plot(range(1, len(myPerceptronClassifier.errors_)+1), myPerceptronClassifier.errors_, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.draw()

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    # Uncomment this to make your Perceptron evaluated
    perceptronPred = myPerceptronClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    #print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    #evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    evaluator.printComparison(data.testSet, perceptronPred)
    # Uncomment this to make your Perceptron evaluated
    evaluator.printAccuracy(data.testSet, perceptronPred)

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)
    plt.show()
if __name__ == '__main__':
    main()
