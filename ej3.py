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
        self.layers = [Layer(params[i], i) for i in range(len(params))]
        self.min_error = sys.maxsize
        self.max_iter = max_iter

    def predict(self, data_input, best=True):
        results = []
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.process(next_layer_input, best)
            results.append(next_layer_input)
        return results[-1]
    
    def process(self, data_input):
        return self.predict(data_input, False)

    def train(self, data_input, expected_output):
        iteration = 0
        while self.min_error > 0.1 and iteration < self.max_iter:
            mu = np.random.randint(0, len(data_input))

            results = []
            next_layer_input = data_input[mu]
            for layer in self.layers:
                next_layer_input = layer.process(next_layer_input)
                results.append(next_layer_input)
                # OK
            
           
            
            next_layer_deltas, next_layer_weights = self.layers[-1].train_last_layer(expected_output[mu], results[-1])

            for i, layer in enumerate(list(reversed(self.layers))[:-1]):
                next_layer_deltas, next_layer_weights = layer.train(results[layer.id - 1], next_layer_deltas, next_layer_weights)

            error = sum((expected_output[mu] - self.process(data_input[mu]))**2 for mu in range(0, len(expected_output)))/2

            if error < self.min_error:
                self.min_error = error
                for layer in self.layers:
                    layer.set_min_weights()
            iteration += 1
        return self.min_error
    
def binary_crossentropy(expected_output, result):
    epsilon = 1e-10
    result = np.clip(result, epsilon, 1 - epsilon)
    return -expected_output * np.log(result + epsilon) - (1 - expected_output) * np.log(1 - result + epsilon)

class Layer:
    def __init__(self, params: list, id):
        self.neurons = [Neuron(params[i], i, id) for i in range(len(params))]
        self.id = id

    def set_min_weights(self):
        for neuron in self.neurons:
            neuron.min_weights = neuron.weights
            neuron.min_bias = neuron.bias

    def process(self, data_input, best=False):
        return [neuron.predict(data_input, best) for neuron in self.neurons]

    def train_last_layer(self, expected_output, value):
        layer_deltas = []
        layer_weights = []
        for i, neuron in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train_last_layer(expected_output, value)
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        
        return layer_deltas, layer_weights

    def train(self, data_input, next_layer_deltas, next_layer_weights):
        layer_deltas = []
        layer_weights = []
        for i, neuron in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train(data_input, next_layer_deltas, next_layer_weights)
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        return layer_deltas, layer_weights


class Neuron:
    def __init__(self, params: NeuronParams, id, layer_id=None):
        self.weights = np.random.rand(params.dimensions)
        self.id = id
        self.layer_id = layer_id
        self.min_weights = None
        self.bias = np.random.rand()
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
        'step': lambda x, b: 1,
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2)
    }        

    def compute_excitement(self, data_input, best=False):
        weights = self.min_weights if best else self.weights
        bias = self.min_bias if best else self.bias
        return sum(data_input * weights) + bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)

    def predict(self, data_input, best=False):
        return self.compute_activation(self.compute_excitement(data_input, best))
    
    def calculate_delta_weights(self, data_input, next_layer_deltas, next_layer_weights):
        excitement = self.compute_excitement(data_input)
        aux = sum(next_layer_deltas[i] * next_layer_weights[i][self.id] for i in range(len(next_layer_deltas)))
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = aux * derivative
        return self.learning_rate * delta, delta
    
    def train_last_layer(self, expected_output, value):
        excitement = self.compute_excitement(value)
        activation = self.compute_activation(excitement)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = (expected_output - activation) * derivative
        delta_weights = []
        for i in range(len(self.weights)):
            delta_weights.append(self.learning_rate * delta * value[0])

        self.weights = self.weights + delta_weights
        self.bias = self.bias + self.learning_rate * delta
        return delta, self.weights

    def train(self, data_input, next_layer_delta, next_layer_weights):
        delta_weight, delta = self.calculate_delta_weights(data_input, next_layer_delta, next_layer_weights)
        delta_weights = []
        for i in range(len(self.weights)):
            delta_weights.append(self.learning_rate * delta * data_input[i])

        self.weights = self.weights + delta_weights
        self.bias = self.bias + self.learning_rate * delta
        return delta, self.weights

def main():
    # data = pd.read_csv('TP3-ej2-conjunto.csv')

    # example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    # example_data_output = data['y'].values

    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_input = np.insert(example_data_input, 0, 1, axis=0)
    example_data_output = np.array([1, 1, -1, -1])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 0.01, 'linear', 0) for _ in range(3)]
    layer_two = [NeuronParams(len(layer_one), 0.1, 'sigmoid', 1) for _ in range(2)]
    layer_three = [NeuronParams(len(layer_two), 0.01, 'tan_h', 0.5) for _ in range(1)]
    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 1000)
    min_error = neural_network.train(example_data_input, example_data_output)
    print(f'Min error: {min_error}')


    result_1 = neural_network.predict(example_data_input[0])
    result_2 = neural_network.predict(example_data_input[1])
    result_3 = neural_network.predict(example_data_input[2])
    result_4 = neural_network.predict(example_data_input[3])
    # result_5 = tree.predict([1, 4, 3])
    print(f'Result 1: {result_1} - expected: {example_data_output[0]}')
    print(f'Result 2: {result_2} - expected: {example_data_output[1]}')
    print(f'Result 3: {result_3} - expected: {example_data_output[2]}')
    print(f'Result 4: {result_4} - expected: {example_data_output[3]}')
    # print(f'Result 5: {result_5}')

    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])
    
    cp = ax.contourf(X, Y, Z, levels=1,  cmap='coolwarm')
    plt.colorbar(cp)

    ax.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

    plt.show()

if __name__ == "__main__":
    main()