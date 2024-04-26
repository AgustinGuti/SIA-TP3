import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class NeuronParams:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta
   
class NeuralNetwork:
    def __init__(self, params: list, max_iter=1000):
        self.layers = [Layer(params[i]) for i in range(len(params))]
        self.min_error = sys.maxsize
        self.max_iter = max_iter

    def train(self, data_input, expected_output):
        i = 0
        while self.min_error > 0.1 and i < self.max_iter:
            mu = np.random.randint(0, len(data_input))

            forward_results = []
            next_layer_input = data_input[mu]
            for layer in self.layers:
                next_layer_input = layer.process(next_layer_input)
                forward_results.append(next_layer_input)

            next_layer_delta, next_layer_weights = self.layers[-1].train_last_layer(expected_output, forward_results[-1])

            for layer, i in enumerate(reversed(self.layers)[:-1]):
                next_layer_delta, next_layer_weights = layer.train(forward_results[-i], next_layer_delta, next_layer_weights)

            
            error = sum(sum((expected_output[mu] - forward_result[mu])**2 for mu in range(0, len(data_input)))/2 for forward_result in forward_results)
            if error < self.min_error:
                self.min_error = error
                for layer in self.layers:
                    layer.set_min_weights()
            i += 1
        return self.min_error

class Layer:
    def __init__(self, params: list):
        self.neurons = [Neuron(params[i]) for i in range(len(params))]

    def set_min_weights(self):
        for neuron in self.neurons:
            neuron.min_weights = neuron.weights

    def process(self, data_input):
        return [neuron.predict(data_input) for neuron in self.neurons]

    def train_last_layer(self, expected_output, value):
        layer_deltas = []
        layer_weights = []
        for neuron, i in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train_last_layer(expected_output[i], value)
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        
        return layer_deltas, layer_weights

    def train(self, data_input, next_layer_delta, next_layer_weights):
        layer_deltas = []
        layer_weights = []
        for neuron, i in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train(data_input, next_layer_delta[i], next_layer_weights[i])
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        return layer_deltas, layer_weights


class Neuron:
    def __init__(self, params: NeuronParams):
        self.weights = np.random.rand(params.dimensions)
        self.min_weights = None
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
        return sum(value * self.weights)

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)

    def predict(self, data_input):
        return [self.compute_activation(self.compute_excitement(value)) for value in data_input]
    
    def calculate_delta_weights(self, data_input, next_layer_delta, next_layer_weights):
        excitement = self.compute_excitement(data_input)
        aux = sum(next_layer_delta * next_layer_weights)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = aux * derivative
        return self.learning_rate * delta * data_input, delta
    
    def train_last_layer(self, expected_output, value):
        excitement = self.compute_excitement(value)
        activation = self.compute_activation(excitement)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = (expected_output - activation) * derivative
        self.weights = self.weights + self.learning_rate * delta * value
        return delta, self.weights

    def train(self, data_input, next_layer_delta, next_layer_weights):
        delta_weights, delta = self.calculate_delta_weights(data_input, next_layer_delta, next_layer_weights)
        self.weights = self.weights + delta_weights
        return delta, self.weights

def calculate_error(data, expected_output):
    return sum([abs(expected_output[mu] - data[mu])**2 for mu in range(0, len(data))])/2

def main():
    # data = pd.read_csv('TP3-ej2-conjunto.csv')

    # example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    # example_data_output = data['y'].values

    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    example_data_input = np.insert(example_data_input, 0, 1, axis=0)
    example_data_output = np.array([1, 1, -1, -1])

    dimensions = len(example_data_input[0])
    # Creates a neural network with 2 layers, the first one with 3 neurons and the second one with 10 neurons.
    # The first layer has 3 neurons with linear activation function and the second layer has 10 neurons with step activation function.
    layer_one = [NeuronParams(dimensions, 0.1, 'linear', 0) for _ in range(3)]
    layer_two = [NeuronParams(dimensions, 0.1, 'step', 0) for _ in range(10)]
    neurons_params = [layer_one, layer_two]
    neural_network = NeuralNetwork(neurons_params, 1000)
    min_error = neural_network.train(example_data_input, example_data_output)
    print(f'Min error: {min_error}')

if __name__ == "__main__":
    main()