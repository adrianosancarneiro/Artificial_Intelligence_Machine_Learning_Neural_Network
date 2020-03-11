import math
import numpy as np
from numpy import exp, array, random, dot
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing
from decimal import *
import matplotlib.pyplot as plt


class Util:
    def __init__(self):
        self.data = None

    def Normalise_Data(self, data):
        return preprocessing.scale(data)
        # return np.array([a - data.mean()/data.std() for a in data])
        # return np.array([a / data.max() for a in data])
        # return np.array([(a - data.min())/ (data.max() - data.min()) for a in data])

class Layer:
    def __init__(self):
        self.id = 0
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def compute_output_fired_neuron(self):
        higher_output_value = 0
        fired_neuron = None
        for neuron in self.neurons:
            if neuron.a > higher_output_value:
                higher_output_value = neuron.a
                fired_neuron = neuron
        return fired_neuron.target

class Neuron():
    def __init__(self):
        self.id = 0
        self.attributes = []
        self.weights = []
        self.h = 0
        self.a = 0
        self.error = 0
        self.target = None
        self.fired = False

    def add_random_weight(self):
        self.weights.append(np.random.uniform(low=-3, high=3))

    def compute_H(self):
        # np.dot returns sum of the multiplication matrix of neurons and attributes
        # (w1 * a1) + (w2 * a2) ... but it does it at once
        self.h = np.dot(self.weights, self.attributes)

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def compute_a(self, deriv=False):
        # The derivative of the Sigmoid function.
        # This is the gradient of the Sigmoid curve.
        # It indicates how confident we are about the existing weight.
        if(deriv == True):
            self.a = (self.h*(1-self.h))
        self.a = 1 / (1 + np.exp(- self.h))

    def compute_a_tanh_(self, x):
        z = (2 / (1 + np.exp(-2 * x))) - 1
        self.a = z

    def compute_error_output_layer(self, input_training_target):
        if(input_training_target == self.target):
            target_error = 1
        else:
            target_error = 0
        self.error = self.a * (1 - self.a) * (self.a - target_error)

    def compute_error_hiden_layer(self, previous_layer_neuron_weights, previous_layer_neuron_errors):
        self.error = self.a * (1 - self.a) * (np.dot(previous_layer_neuron_weights, previous_layer_neuron_errors))

    def compute_new_weights(self, n):
        weight_index = 0
        for weight, attribute in zip(self.weights, self.attributes):
            self.weights[weight_index] = weight - n * self.error * attribute
            weight_index += 1

class Neural_Network():

    def __init__(self):
        self.layers = []
        self.bias_term = None

    # Create the structure with certain number of layers and its respective neurons
    # receive an array of number eg: [2, 3, 4] => 3 layer the 1st have 2 neurons
    # 2nd have 3 neurons and the 3rd have 4 neurons
    def add_layers_neurons(self, layers_neurons_number):
        layer_id = 1
        for neurons_number in layers_neurons_number:
            layer = Layer()
            layer.id = layer_id
            neuron_id = 1
            for _ in range(neurons_number):
                n = Neuron()
                n.id = neuron_id
                layer.add_neuron(n)
                neuron_id += 1
            self.layers.append(layer)
            layer_id += 1

    # Make the neural netword learn based on the data an targets inputed
    def training(self, data, targets, layers_neurons_number, bias_term, epochs, n, validation_data, validation_target):
        self.add_layers_neurons(layers_neurons_number)
        self.compute_output_neuron_target(targets)
        self.add_bias_term(bias_term)
        self.data_row_accuracy_list = []
        self.each_epoch_accuracy_list = []

        # output_n1 = []
        # output_n2 = []
        epochs_index = 0
        for _ in range(epochs):
            # output_n1.append("New apoch")
            # output_n2.append("New apoch")
            input_training_index = 0
            for input_training_values, input_training_target in zip(data, targets):
                self.feed_forward(input_training_values, input_training_target)
                #change the weights
                if input_training_index != 0:
                    self.backpropagation(n)
                # r = 0
                # output_n1.append(str(self.layers[-1].neurons[0].a) + " | " + str(self.layers[-1].neurons[0].target))
                # output_n2.append(str(self.layers[-1].neurons[1].a) + " | " + str(self.layers[-1].neurons[1].target))
                # output_n1.append(str(self.layers[-1].neurons[0].a))
                # output_n2.append(str(self.layers[-1].neurons[1].a))
                input_training_index += 1
                self.clean()
                # for neuron in self.layers[-1].neurons:
                #     print neuron.a
                #     print neuron.target
                if epochs_index == 0:
                    each_row_accuracy = self.validation(validation_data, validation_target)
                    self.data_row_accuracy_list.append(each_row_accuracy)
            # create a graphic using self.data_row_accuracy_list
            # self.data_row_accuracy_list = []
            each_epoch_accuracy = self.validation(validation_data, validation_target)
            self.each_epoch_accuracy_list.append(each_epoch_accuracy)
            epochs_index += 1
        # create a graphic using self.each_epoch_accuracy_list
        # self.each_epoch_accuracy_list = []
        # for n1 in output_n1:
        #     print n1
        # for n2 in output_n2:
        #     print n2
        x=2

    def clean(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.h = None
                neuron.a = None
                neuron.error = None
                del neuron.attributes[-(len(neuron.attributes)-1):]

    def validation(self, validation_data, validation_target):
        validation_predicted = self.prediction(validation_data)
        accuracy = self.compute_accuracy(validation_predicted, validation_target)
        return accuracy

    def prediction(self, data):
        predicted_values = []
        for input_predicting_values in data:
            layers_index = 0
            for layer in self.layers:
                for neuron in layer.neurons:
                    if layers_index == 0:
                        for input_value in input_predicting_values:
                            if (len(input_predicting_values)) > len(neuron.attributes)-1:
                                neuron.attributes.append(input_value)
                    else:
                        for previous_layer_neuron in self.layers[layers_index - 1].neurons:
                            if (len(self.layers[layers_index - 1].neurons)) > len(neuron.attributes)-1:
                                neuron.attributes.append(previous_layer_neuron.a)
                    neuron.compute_H()
                    neuron.compute_a()
                layers_index += 1
            layers_index = 0
            predicted_value = self.layers[-1].compute_output_fired_neuron()
            predicted_values.append(predicted_value)
            self.clean()
        return predicted_values

    def compute_accuracy(self, predicted_values, target_values):
        right_prediction = 0
        for predicted_value, target_value in zip(predicted_values, target_values):
            if predicted_value == target_value:
                right_prediction += 1
        accuracy = float(Decimal(right_prediction) / Decimal(len(predicted_values)))
        return accuracy

    # Decide which output neuron will relate to each unique target
    def compute_output_neuron_target(self, targets):
        unique_target_values = set(targets)
        if(len(self.layers[-1].neurons) != len(unique_target_values)):
            # print "The last layer have to contain {} neurons, insted of {}".format(len(unique_target_values), len(self.layers[-1].neurons))
            raise Exception("The last layer have to contain {} neurons, insted of {}".format(len(unique_target_values), len(self.layers[-1].neurons)))
        else:
            for neuron, target in zip(self.layers[-1].neurons, unique_target_values):
                neuron.target = target

    def backpropagation(self, n):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.compute_new_weights(n)

    # Compute the h, a and errors.
    # If there are no weights, random ones will be added.
    # If there are some we will keep them
    def feed_forward(self, input_training_values, input_training_target):
        layers_index = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                if layers_index == 0:
                    for input_value in input_training_values:
                        # if I have more input training values than weights
                        # that means I miss some weights. Have to have the same inputs and weights number
                        if (len(input_training_values)) > len(neuron.weights)-1:
                            neuron.add_random_weight()

                        if (len(input_training_values)) > len(neuron.attributes)-1:
                            neuron.attributes.append(input_value)
                else:
                    for previous_layer_neuron in self.layers[layers_index - 1].neurons:
                        # Adding +1 to the numbers of previous layer neurons verification
                        # because the bayes term is plus the neurons is the amount of
                        # weights I will need in the current neuron
                        if (len(self.layers[layers_index - 1].neurons)) > len(neuron.weights)-1:
                            neuron.add_random_weight()

                        if (len(self.layers[layers_index - 1].neurons)) > len(neuron.attributes)-1:
                            neuron.attributes.append(previous_layer_neuron.a)
                neuron.compute_H()
                neuron.compute_a()
            layers_index += 1
        layers_index = 0

        # Compute the error terms
        for layer in reversed(self.layers):
            neurons_index = 0
            for neuron in layer.neurons:
                if layers_index == 0:
                    neuron.compute_error_output_layer(input_training_target)
                else:
                    previous_layer_neuron_weights = []
                    previous_layer_neuron_errors = []
                    for previous_layer_neuron in self.layers[layers_index].neurons:
                        # neuron_index + 1 is needed because the top weight in the previous layer neuron is the bias weight
                        previous_layer_neuron_weights.append(previous_layer_neuron.weights[neurons_index + 1])
                        previous_layer_neuron_errors.append(previous_layer_neuron.error)
                    neuron.compute_error_hiden_layer(previous_layer_neuron_weights, previous_layer_neuron_errors)
                neurons_index += 1
            layers_index += 1

    # Adding the bias term as an attribute for all the node and layers
    # Make sure you add the bias term when all the layers is alread there.
    def add_bias_term(self, bias_term_value):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.attributes.append(bias_term_value)
                neuron.add_random_weight()
        # self.bias_term = bias_term_value

    def Compute_Fire(self, neuron):
        if neuron.a > 0:
            return True
        else:
            return False

    def create_first_data_iteration_accuracy(self):
        plt.plot(self.data_row_accuracy_list, 'b')
        plt.legend()
        plt.show()  
    
    def create_all_epoch_accuracy(self):
        plt.plot(self.each_epoch_accuracy_list, 'b')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #############################################
    ## BREAST CANCER DATASET
    #############################################
    print("BREAST CANCER DATASET")

    names = ["id",
             "clump Thickness",
             "unif_cell_size",
             "unif_cell_shape",
             "marg_adhesion",
             "single_epith",
             "bare_nuclei",
             "bland_chrom",
             "normal_nucleoli",
             "mitoses",
             "class"]

    df = pd.read_csv(
        '/Users',
        names=names
    )

    # -99999 the algorith consider this number as a outlier
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    nn2 = Neural_Network()
    u2 = Util()
    #df = u2.Normalise_Data(df)

    data2 = np.array(df.drop(['class'], 1))

    target2 = np.array(df['class'])

    data_train2, data_test2, target_train2, target_test2 = train_test_split(data2, target2,
                                                                        train_size=0.8,
                                                                        test_size=0.2,
                                                                        random_state=123)

    data_train2, data_validation2, target_train2, target_validation2 = train_test_split(data_train2, target_train2,
                                                                                    train_size=0.8,
                                                                                    test_size=0.2,
                                                                                    random_state=123)

    data_train2 = u2.Normalise_Data(data_train2)
    data_validation2 = u2.Normalise_Data(data_validation2)
    data_test2 = u2.Normalise_Data(data_test2)

    target_unique_values2 = set(target_train2)

    layers_neurons_number2 = [4, len(target_unique_values2)]

    print "configurations (# of layers, # of nodes per layer) = " + str(layers_neurons_number2)

    nn2.training(data_train2, target_train2, layers_neurons_number2, -1, 150, 0.2, data_validation2, target_validation2)

    nn2.create_first_data_iteration_accuracy()

    nn2.create_all_epoch_accuracy()

    data_test_prediction2 = nn2.prediction(data_test2)
    data_test_accuracy2 = nn2.compute_accuracy(data_test_prediction2, target_test2)

    print "testing data accuracy = " + str(data_test_accuracy2)


    #############################################
    ## Iris Dataset
    #############################################
    print("Iris Dataset")
    nn = Neural_Network()
    u = Util()

    iris=load_iris()

    data,target = iris.data,iris.target
    #data = u.Normalise_Data(data)
    data_train, data_test, target_train, target_test = train_test_split(    data, target,
                                                                            train_size=0.8,
                                                                            test_size=0.2,
                                                                            random_state=123)

    data_train, data_validation, target_train, target_validation = train_test_split(    data_train, target_train,
                                                                                        train_size=0.8,
                                                                                        test_size=0.2,
                                                                                        random_state=123)

    data_train = u.Normalise_Data(data_train)
    data_validation = u.Normalise_Data(data_validation)
    data_test = u.Normalise_Data(data_test)

    layers_neurons_number = [3, 3]

    print "configurations (# of layers, # of nodes per layer) = " + str(layers_neurons_number)

    nn.training(data_train, target_train, layers_neurons_number, -1, 200, 0.4, data_validation, target_validation)

    nn.create_first_data_iteration_accuracy()
    
    nn.create_all_epoch_accuracy()



    data_test_prediction = nn.prediction(data_test)
    data_test_accuracy = nn.compute_accuracy(data_test_prediction, target_test)

    print "testing data accuracy = " + str(data_test_accuracy)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    # train_data = array([  [0, 0, 1],
    #                 [1, 1, 1],
    #                 [1, 0, 1],
    #                 [0, 1, 1]])
    # train_targets = array([0, 1, 1, 0]).T

    # validation_data = array([[1, 1, 0],
    #                         [0, 0, 1],
    #                         [0, 1, 0],
    #                         [1, 0, 0]])
    # validation_targets = array([1, 0, 0, 1]).T

    # nn1 = Neural_Network()

    # layers_neurons_number = [10, 2]

    # nn1.training(train_data, train_targets, layers_neurons_number, -1, 200, 0.1, validation_data, validation_targets)

    #WORKING WELL
    # train_data = array([    [1, 1],
    #                         [1, 0],
    #                         [0, 1],
    #                         [0, 0]])
    # train_targets = array(["A", "A", "A", "B"]).T

    # nn1 = Neural_Network()

    # layers_neurons_number = [3, 2]

    # nn1.training(train_data, train_targets, layers_neurons_number, -1, 150, 0.5, train_data, train_targets)

   #WORKING WELL
    # train_data = array([    [1, 1],
    #                         [1, 0],
    #                         [0, 1],
    #                         [0, 0]])
    # train_targets = array(["A", "A", "A", "B"]).T
    #
    # nn1 = Neural_Network()
    #
    # layers_neurons_number = [2, 2]
    #
    # nn1.training(train_data, train_targets, layers_neurons_number, -1, 150, 0.8, train_data, train_targets)

    #DOES NOT WORKING - DOES NOT LEARN WITH ONLY ONE HIDEN NEURON
    # train_data = array([    [1, 1],
    #                         [1, 0],
    #                         [0, 1],
    #                         [0, 0]])
    # train_targets = array(["A", "A", "A", "B"]).T

    # nn1 = Neural_Network()

    # layers_neurons_number = [1, 2]

    # nn1.training(train_data, train_targets, layers_neurons_number, -1, 150, 1.0, train_data, train_targets)







    data = array([  [1.2, -0.2],
                    [0.8, 0.1]])

    targets = array([0, 1]).T



    #Intialise a single neuron neural network.

    # nn = Neural_Network()

    # u = Util()

    # d1 = u.Normalise_Data(data)

    # layers_neurons_number = [2, 2]

    # nn.training(data, targets, layers_neurons_number, -1, 1000, 1.0, )

    # data_test_prediction = nn.prediction(data_test)
    # data_test_accuracy = nn.compute_accuracy(data_test_prediction, target_test)

