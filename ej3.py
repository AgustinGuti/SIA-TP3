import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class NeuronParams:
    def __init__(self, dimensions, max_iter, learning_rate=0.1, activation_function='linear', beta=100):
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta

class NeuralNetwork:
    def __init__(self, params: list):
        self.layers = [Layer(params[i]) for i in range(len(params))]

    def process(self, data_input):
        for layer in self.layers:
            data_input = layer.process(data_input)
        return data_input

    def train(self, data_input, expected_output):
        # TODO
        pass

class Layer:
    def __init__(self, params: list):
        self.neurons = [Neuron(params[i]) for i in range(len(params))]

    def process(self, data_input):
        return [neuron.compute_activation(neuron.compute_excitement(data_input)) for neuron in self.neurons]

    def train(self, data_input, expected_output):
        # TODO
        pass


class Neuron:
    def __init__(self, params: NeuronParams):
        self.weights = np.random.rand(params.dimensions)
        self.bias = np.random.rand()
        self.error = None
        self.min_error = sys.maxsize
        self.max_iter = params.max_iter
        self.min_weights = None
        self.min_bias = None
        self.learning_rate = params.learning_rate
        self.activation_function = params.activation_function
        self.beta = params.beta

    activation_functions = {
        'step': lambda x, b: 1 if x >= 0 else -1,
        'linear': lambda x, b=0: x,
        'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
        'tan_h': lambda x, b: np.tanh(b*x)
    }

    derivative_activation_functions = {
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2)
    }        

    def compute_excitement(self, value):
        return sum(value * self.weights) + self.bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)
    
    def train(self, data_input, expected_output):
        # TODO
        pass

def calculate_error(data, expected_output):
    return sum([abs(expected_output[mu] - data[mu])**2 for mu in range(0, len(data))])/2

def main():
    data = pd.read_csv('TP3-ej2-conjunto.csv')

    example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    example_data_output = data['y'].values

    dimensions = len(example_data_input[0])
    # Creates a neural network with 2 layers, the first one with 3 neurons and the second one with 10 neurons.
    # The first layer has 3 neurons with linear activation function and the second layer has 10 neurons with step activation function.
    neurons_params = [[NeuronParams(1000, dimensions, 0.1, 'linear', 0) for i in range(3)], [NeuronParams(1000, dimensions, 0.1, 'step', 0) for i in range(10)]]
    neural_network = NeuralNetwork(neurons_params)

if __name__ == "__main__":
    main()