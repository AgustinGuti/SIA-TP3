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

    def get_weights(self):
        return [[(neuron.weights, neuron.bias) for neuron in layer.neurons] for layer in self.layers]
    
    def set_min_weights(self, weights):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.min_weights = weights[i][j][0]
                neuron.min_bias = weights[i][j][1]

    def print_weights(self):
        for layer in self.layers:
            print(f'Layer: {layer.id}')
            for neuron in layer.neurons:
                print(f'Neuron: {neuron.id} - Weights: {neuron.weights} - Bias: {neuron.bias}')

    def train(self, data_input, expected_output):
        iteration = 0
        best_weights_history = []
        error_history = []
        while self.min_error > 1e-3 and iteration < self.max_iter:
            mu = np.random.randint(0, len(data_input))

            results = []
            next_layer_input = data_input[mu]
            for layer in self.layers:
                next_layer_input = layer.process(next_layer_input)
                results.append(next_layer_input)
            
            next_layer_deltas, next_layer_weights = self.layers[-1].train_last_layer(expected_output[mu], results[-2])

            for layer in list(reversed(self.layers))[1:]:
                training_input = results[layer.id-1]
                if layer.id == 0:
                    training_input = data_input[mu]
                next_layer_deltas, next_layer_weights = layer.train(training_input, next_layer_deltas, next_layer_weights)

                
            error = sum(sum((expected_output[mu] - self.process(data_input[mu]))**2 for mu in range(0, len(expected_output)))/2)

            improved = ''
            if error < self.min_error:
                error_history.append(error)
                best_weights_history.append(self.get_weights())
                self.min_error = error
                for layer in self.layers:
                    layer.set_min_weights()
                improved = '- True'

            print(f'Iteration: {iteration} - Error: {error} - Min Error: {self.min_error} {improved}')

            iteration += 1
        return self.min_error, iteration, best_weights_history, error_history

class Layer:
    def __init__(self, params: list, id):
        self.neurons = [Neuron(params[i], i, id) for i in range(len(params))]
        self.id = id

    def set_min_weights(self):
        for neuron in self.neurons:
            neuron.min_weights = neuron.weights
            neuron.min_bias = neuron.bias

    def process(self, data_input, best=False):
        arr = np.array([neuron.predict(data_input, best) for neuron in self.neurons])
        return arr

    def train_last_layer(self, expected_output, last_layer_output):
        layer_deltas = []
        layer_weights = []

        for i, neuron in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train_last_layer(expected_output[i], last_layer_output)
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

class MomentumParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_m = 0

    def get_m(self, iteration, derivatives):
        m = self.beta * self.last_m + (1 - self.beta) * derivatives
        self.last_m = m
        return m / (1 - self.beta**iteration)
    
class RMSPropParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_v = 0

    def get_v(self, iteration, gradient):
        v = self.beta * self.last_v + (1 - self.beta) * gradient**2
        self.last_v = v
        return v / (1 - self.beta**iteration)
    
    def get_delta(self, iteration, gradient):
        return gradient / np.sqrt(self.get_v(iteration, gradient) + 1e-8)

class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.momentum_params = MomentumParams(beta1)
        self.rms_prop_params = RMSPropParams(beta2)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _get_m(self, iteration, derivatives):
        return self.momentum_params.get_m(iteration, derivatives)

    def _get_v(self, iteration, derivatives):
        return self.rms_prop_params.get_v(iteration, derivatives)
        
    def get_delta(self, iteration, derivatives):
        m = self._get_m(iteration, derivatives)
        v = self._get_v(iteration, derivatives)
        return m / (np.sqrt(v + self.epsilon)) # TODO check if it should be v or v + epsilon
    

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
        self.adam_params = AdamParams()
        self.momentum_params = MomentumParams()
        self.rms_prop_params = RMSPropParams()
        self.iteration = 1
        self.optimizer = 'a'

    activation_functions = {
        'linear': lambda x, b=0: x,
        'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
        'tan_h': lambda x, b: np.tanh(b*x),
    }

    derivative_activation_functions = {
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2),
    }        

    def compute_excitement(self, data_input, best=False):
        weights = self.min_weights if best else self.weights
        bias = self.min_bias if best else self.bias
        return np.dot(data_input, weights) + bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)

    def predict(self, data_input, best=False):
        return self.compute_activation(self.compute_excitement(data_input, best))
    
    def calculate_delta_weights(self, data_input, next_layer_deltas, next_layer_weights):
        excitement = self.compute_excitement(data_input)
        next_layer_weights = np.array(next_layer_weights)
        aux = np.sum(np.multiply(next_layer_deltas, next_layer_weights[:, self.id]))

        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)

        error_gradient = np.multiply(aux * derivative, data_input)

        # Shoult it be aux?
        if self.optimizer == 'adam':
            delta = self.adam_params.get_delta(self.iteration, error_gradient)
        elif self.optimizer == 'momentum':
            delta = self.momentum_params.get_m(self.iteration, error_gradient)
        elif self.optimizer == 'rms':
            delta = self.rms_prop_params.get_delta(self.iteration, error_gradient)
        else:
            delta = aux * derivative

        # print(f'Iteration: {self.iteration} - Delta: {delta} - Error Gradient: {error_gradient} - Layer: {self.layer_id}')
        return np.multiply(self.learning_rate, delta), aux*derivative
    
    def train_last_layer(self, expected_output, previous_layer_output):
        excitement = self.compute_excitement(previous_layer_output)
        activation = self.compute_activation(excitement)

        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)

        error_gradient = -np.multiply((expected_output - activation) * derivative, previous_layer_output)

        if self.optimizer == 'adam':
            delta = self.adam_params.get_delta(self.iteration, error_gradient)
            self.weights = self.weights + self.learning_rate * delta  
        elif self.optimizer == 'momentum':
            delta = self.momentum_params.get_m(self.iteration, error_gradient)
            self.weights = self.weights + delta * self.learning_rate
        elif self.optimizer == 'rms':
            delta = self.rms_prop_params.get_delta(self.iteration, error_gradient)
            self.weights = self.weights + delta * self.learning_rate
        else:
            delta = (expected_output - activation) * derivative
            delta_weights = np.multiply(self.learning_rate * delta, previous_layer_output)
            self.weights = self.weights + delta_weights

        self.bias = self.bias + self.learning_rate * (expected_output - activation) * derivative
        self.iteration += 1
        return delta, self.weights

    def train(self, data_input, next_layer_delta, next_layer_weights):
        delta_weight, delta = self.calculate_delta_weights(data_input, next_layer_delta, next_layer_weights)

        if self.optimizer == 'adam' or self.optimizer == 'momentum' or self.optimizer == 'rms':
            self.weights = self.weights + delta_weight 
        else:
            delta_weights = []

            for i in range(len(self.weights)):
                delta_weights.append(delta_weight * data_input[i])
            self.weights = self.weights + delta_weights

        self.bias = self.bias + delta * self.learning_rate
        self.iteration += 1
        return delta, self.weights

def main():
    with open('TP3-ej3-digitos.txt', 'r') as f:
        lines = [line.split() for line in f]
    data = [list(map(int, [num for sublist in lines[i:i+7] for num in sublist])) for i in range(0, len(lines), 7)]

    example_data_input = data

    # example_data_input = np.insert(example_data_input, 0, 1, axis=0)
    example_data_output = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    # example_data_output = np.array([[i] for i in range(10)])
    
    # example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_output = np.array([[-1], [-1], [-1], [1]])

    # example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_output = np.array([[1], [1], [-1], [-1]])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 0.01, 'linear', 0.2) for _ in range(50)]
    layer_two = [NeuronParams(len(layer_one), 0.01,  'tan_h', 0.2) for _ in range(5)]
    layer_three = [NeuronParams(len(layer_two), 0.01, 'tan_h', 0.5) for _ in range(10)]
    # layer_four = [NeuronParams(len(layer_three), 0.001, 'linear', 0.5) for _ in range(1)]

    # example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_output = np.array([[1], [1], [-1], [-1]])

    # dimensions = len(example_data_input[0])
    # layer_one = [NeuronParams(dimensions, 1, 'linear', 0.1) for _ in range(1)]
    # layer_two = [NeuronParams(len(layer_one), 1, 'tan_h', 0.2) for _ in range(5)]
    # layer_three = [NeuronParams(len(layer_two), 1, 'tan_h', 0.5) for _ in range(1)]

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 1000)

    min_error, iterations, best_weights_history, error_history = neural_network.train(example_data_input, example_data_output)
    # neural_network.print_weights()
    print(f'Min error: {min_error} - Iterations: {iterations}')

    results = []
    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data)
        result = [round(res, 2) for res in result]
        print(f'Result {i}: {result} - expected: {example_data_output[i]}')
        results.append(result)

#     fig, axs = plt.subplots(len(results), figsize=(10, 30))

# # For each input, create a bar chart of the result and expected result
#     for i, (result, expected) in enumerate(zip(results, example_data_output)):
#         axs[i].bar(np.arange(len(result)), result, alpha=0.7, label='Result')
#         axs[i].bar(np.arange(len(expected)), expected, alpha=0.7, label='Expected')
#         axs[i].set_title(f'Input {i+1}')
#         axs[i].legend()

#     plt.tight_layout()
#     plt.show()

    # for weights in best_weights_history:
    #     neural_network.set_min_weights(weights)
    #     for i, input_data in enumerate(example_data_input):
    #         result = neural_network.predict(input_data)
    #         result = [round(res, 2) for res in result]
    #         print(f'Result {i}: {result} - expected: {example_data_output[i]}')

if __name__ == "__main__":
    main()
