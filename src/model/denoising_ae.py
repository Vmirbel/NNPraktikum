# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder
from util.loss_functions import MeanSquaredError

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

        self.error_function = MeanSquaredError()

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # First hidden layer
        number_of_1st_hidden_layer = 100

        self.layers.append(LogisticLayer(train.input.shape[1],
                                         number_of_1st_hidden_layer, None,
                                         activation="sigmoid",
                                         is_classifier_layer=False))

        # Output layer
        self.layers.append(LogisticLayer(number_of_1st_hidden_layer,
                                         785, None,
                                         activation="sigmoid",
                                         is_classifier_layer=False))

        self.training_set.noise_input = self._add_noise(self.training_set.input)

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.training_set.noise_input = np.insert(self.training_set.noise_input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)


    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                error = self.evaluate(self.validation_set)
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(error)
                print("Error on validation: {0:.2f}"
                      .format(error))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for input, target in zip(self.training_set.noise_input,
                              self.training_set.input):
            self._feed_forward(input)
            self._compute_error(target)
            self._update_weights()

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _get_weights(self):
        """
        Get the weights (after training)
        """
        self.layers[0].weights
        pass

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer
        """

        # Feed forward layer by layer
        # The output of previous layer is the input of the next layer
        last_layer_output = inp

        for layer in self.layers:
            last_layer_output = layer.forward(last_layer_output)
            # Do not forget to add bias for every layer
            last_layer_output = np.insert(last_layer_output, 0, 1, axis=0)
        return last_layer_output

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        # Get output layer
        output_layer = self._get_output_layer()

        # Calculate the deltas of the output layer
        output_layer.deltas = target - output_layer.outp

        # Calculate deltas (error terms) backward except the output layer
        for i in reversed(range(0, len(self.layers) - 1)):
            current_layer = self._get_layer(i)
            next_layer = self._get_layer(i+1)
            next_weights = np.delete(next_layer.weights, 0, axis=0)
            next_derivatives = next_layer.deltas

            current_layer.computeDerivative(next_derivatives, next_weights.T)

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        # Update the weights layer by layers
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def evaluate(self, test):
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
        # Once you can classify an instance, just use map for all of the test
        # set.
        noise_input = self._add_noise(test.input)
        err = 0
        for noise_input_curr, target_curr in zip(noise_input, test.input):
            self._feed_forward(noise_input_curr)
            outp_curr = self._get_output_layer().outp
            err += self.error_function.calculate_error(target_curr, outp_curr)
        return err

    def _add_noise(self, inputs):
        noise_size = inputs[0].shape[0]
        zeros_size = int(noise_size * 0.3)
        noise = np.ones(noise_size - zeros_size)
        noise = np.insert(noise, 0, np.zeros(zeros_size))

        noise_inputs = []

        for input_curr in inputs:
            np.random.shuffle(noise)
            noise_inputs.append(input_curr*noise)

        return noise_inputs
