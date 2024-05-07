import numpy as np
import sys
import json
import copy

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

class NeuronParams:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100, optimizer=''):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta
        self.optimizer = optimizer
   
class NeuralNetwork:
    def __init__(self, params: list):
        self.layers = [Layer(params[i], i) for i in range(len(params))]
        self.min_error = sys.maxsize

    def predict(self, data_input, best=True):
        results = []
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.process(next_layer_input, best)
            results.append(next_layer_input)
        return results[-1]
    
    def dump_weights_to_file(self, filename):
        weights = [layer.weights.tolist() for layer in self.layers]
        bias = [layer.biases.tolist() for layer in self.layers]
        data = {
            'weights': weights,
            'biases': bias
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_weights_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(data['weights'][i])
            layer.biases = np.array(data['biases'][i])
            layer.set_min_weights()
    
    def process(self, data_input):
        return self.predict(data_input, False)

    def get_weights(self):
        return [copy.deepcopy(layer.weights) for layer in self.layers]
    
    def get_biases(self):
        return [copy.deepcopy(layer.biases) for layer in self.layers]
    
    def set_min_weights(self, weights, biases=None):
        for i, layer in enumerate(self.layers):
            layer.set_min_weights(weights[i], biases[i] if biases is not None else None)

    def print_weights(self):
        for layer in self.layers:
            print(f'Layer: {layer.id}')
            print(f'Weights: {layer.weights}')

    def train(self, data_input, expected_output, iters):
        iteration = 0
        best_weights_history = []
        best_biases_history = []
        error_history = []
        while self.min_error > 1e-5 and iteration < iters:
            mu = np.random.randint(0, len(data_input))

            results = []
            next_layer_input = data_input[mu]
            for layer in self.layers:
                next_layer_input = layer.process(next_layer_input)
                results.append(next_layer_input)

            gradient = - 2 * (expected_output[mu] - results[-1]) / len(expected_output)

            for layer in reversed(self.layers):
                training_input = results[layer.id-1]
                if layer.id == 0:
                    training_input = data_input[mu]
                gradient = layer.train(training_input, gradient, iteration)
                
            error = calculate_error(self.predict(data_input), expected_output)

            error_history.append(error)
            best_weights_history.append(self.get_weights())
            best_biases_history.append(self.get_biases())

            improved = ''
            if error < self.min_error:
                self.min_error = error
                for layer in self.layers:
                    layer.set_min_weights()
                improved = '- True'

            # print(f'Iteration: {iteration} - Error: {error} - Min Error: {self.min_error} {improved}')

            iteration += 1
        return self.min_error, iteration, best_weights_history, best_biases_history, error_history

def calculate_error(predictions, expected_output):
    return sum(sum((expected_output[i] - predictions[i])**2 for i in range(len(predictions)))/len(predictions))

class Layer:
    def __init__(self, params: list[NeuronParams], id):
        self.id = id
        self.weights = np.array([np.random.rand(params[0].dimensions) for _ in range(len(params))])
        self.biases = [np.random.rand() for _ in range(len(params))]
        self.learning_rate = params[0].learning_rate
        self.min_weights = self.weights
        self.min_biases = self.biases
        self.activation_function = params[0].activation_function
        self.beta = params[0].beta
        self.momentum_params = MomentumParams()
        self.rms_prop_params = RMSPropParams()
        self.adam_params = AdamParams()
        self.optimizer = params[0].optimizer
        
    def set_min_weights(self, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        if biases is not None:
            self.biases = biases

        self.min_weights = self.weights
        self.min_biases = self.biases

    def compute_exitement(self, data_input, best=False):
        weights = self.min_weights if best else self.weights
        biases = self.min_biases if best else self.biases
        return np.dot(data_input, weights.T) + biases
    
    def compute_activation(self, excitement):
        return activation_functions[self.activation_function](excitement, self.beta)

    def process(self, data_input, best=False):
        return self.compute_activation(self.compute_exitement(data_input, best))

    def train(self, training_input, gradient, iteration):
        new_gradient = np.dot(self.weights.T, gradient)

        if self.optimizer == 'momentum':
            change  = self.momentum_params.get_m(iteration, gradient)
        elif self.optimizer == 'rms_prop':
            change = self.rms_prop_params.get_delta(iteration, gradient)
        elif self.optimizer == 'adam':
            change = self.adam_params.get_delta(iteration, gradient)
        else:
            change = gradient

        weights_gradient = np.outer(change, training_input)

        self.weights -= self.learning_rate * weights_gradient
        self.biases -= self.learning_rate * change

        return new_gradient

class MomentumParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_m = 0

    def get_m(self, iteration, gradient):
        m = self.beta * self.last_m + (1 - self.beta) * gradient
        self.last_m = m
        return m # / (1 - self.beta**iteration)
    
class RMSPropParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_v = 0

    def get_v(self, iteration, gradient):
        v = self.beta * self.last_v + (1 - self.beta) * gradient**2
        self.last_v = v
        return v # / (1 - self.beta**iteration)
    
    def get_delta(self, iteration, gradient):
        return gradient / np.sqrt(self.get_v(iteration, gradient) + 1e-8)

class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.momentum_params = MomentumParams(beta1)
        self.rms_prop_params = RMSPropParams(beta2)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _get_m(self, iteration, gradient):
        return self.momentum_params.get_m(iteration, gradient)

    def _get_v(self, iteration, gradient):
        return self.rms_prop_params.get_v(iteration, gradient)
        
    def get_delta(self, iteration, gradient):
        m = self._get_m(iteration, gradient)
        v = self._get_v(iteration, gradient)
        return m / (np.sqrt(v + self.epsilon))
