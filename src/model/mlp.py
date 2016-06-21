
import numpy as np

# from util.activation_functions import Activation
from sklearn.metrics import accuracy_score
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.loss_functions import CrossEntropyError

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 costfunction=CrossEntropyError(), learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

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
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.costfunction = costfunction

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # convert all labels to a 10x1-Vector with "vector[label] == 1"
        #self.training_set.label = list(map(lambda a: self.convert_to_vector(a), self.training_set.label))
        #self.validation_set.label = list(map(lambda a: self.convert_to_vector(a), self.validation_set.label))
        #self.test_set.label = list(map(lambda a: self.convert_to_vector(a), self.test_set.label))

        # add bias values ("1"s) at the beginning of all data sets
        #self.training_set.input = np.insert(self.training_set.input, 0, 1,
        #                                    axis=1)
        #self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
        #                                      axis=1)
        #self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        self.layers = []
        inputs_number = self.training_set.input.shape[1]
        self.layers.append(LogisticLayer(inputs_number, 40, None, "sigmoid", is_classifier_layer=False))
        self.layers.append(LogisticLayer(40, 10, None, output_activation, is_classifier_layer=True))



    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        temp_value = inp

        for layer in self.layers:
            temp_value = layer.forward(temp_value)

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        output_layer = self._get_output_layer()
        output_layer.computeDerivative(target, None)

        for i in reversed(range(0, len(self.layers) - 1)):
            self.layers[i].computeDerivative(self.layers[i+1].deltas, self.layers[i+1].weights)

        return self.costfunction.calculate_error(target, output_layer.outp)

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):

            print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")



    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for inp, label in zip(self.training_set.input,
                              self.training_set.label):
            # convert a label to a vector. Ex.: 7 --> [0,0,0,0,0,0,0,1,0,0]
            label_vector = np.zeros(10)
            label_vector[label] = 1
            self._feed_forward(inp)
            self._compute_error(label_vector)
            self._update_weights()

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        self._feed_forward(test_instance)
        output = self._get_output_layer().outp
        return np.argmax(output)

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def convert_to_vector(self, i):
        """This function could be used to change all labels from a number to a vector.
            Ex.: 7 --> [0,0,0,0,0,0,0,1,0,0]
        """
        arr = np.zeros(10)
        arr[i] = 1
        return arr

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
