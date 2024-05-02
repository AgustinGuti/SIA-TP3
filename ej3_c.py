import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import noise
import json

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
    
    def dump_weights_to_file(self, filename):
        weights = [layer.weights.tolist() for layer in self.layers]
        with open(filename, 'w') as f:
            json.dump(weights, f)

    def load_weights_from_file(self, filename):
        with open(filename, 'r') as f:
            weights = json.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(weights[i])
            layer.set_min_weights()
    
    def process(self, data_input):
        return self.predict(data_input, False)

    def get_weights(self):
        return [layer.weights for layer in self.layers]
    
    def set_min_weights(self, weights):
        for i, layer in enumerate(self.layers):
            layer.set_min_weights(weights[i])

    def print_weights(self):
        for layer in self.layers:
            print(f'Layer: {layer.id}')
            print(f'Weights: {layer.weights}')

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

            output = results[-1]

            # error = np.mean(np.power(expected_output[mu] - output, 2))

            gradient = - 2 * (expected_output[mu] - output) / len(expected_output)

            for layer in reversed(self.layers):
                training_input = results[layer.id-1]
                if layer.id == 0:
                    training_input = data_input[mu]
                gradient = layer.train(training_input, gradient, iteration)
                
            # error = 0
            # for i in range(len(data_input)):
            #     result = self.process(data_input[i])
            #     error += np.mean(np.power(expected_output[i] - result, 2))

            error = sum(sum((expected_output[mu] - self.process(data_input[mu]))**2 for mu in range(0, len(expected_output)))/len(expected_output))

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
    def __init__(self, params: list[NeuronParams], id):
        self.id = id
        self.weights = np.array([np.random.rand(params[0].dimensions) for _ in range(len(params))])
        self.biases = [np.random.rand() for _ in range(len(params))]
        self.learning_rate = params[0].learning_rate
        self.min_weights = None
        self.min_biases = None
        self.activation_function = params[0].activation_function
        self.beta = params[0].beta
        self.momentum_params = MomentumParams()
        self.rms_prop_params = RMSPropParams()
        self.adam_params = AdamParams()
        self.optimizer = ''
        
    def set_min_weights(self, weights=None):
        if weights is not None:
            self.weights = weights
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
    # example_data_output = np.array([[1], [1], [-1], [-1]])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 0.1, 'linear', 0.2) for _ in range(150)]
    layer_two = [NeuronParams(len(layer_one), 0.1,  'tan_h', 0.2) for _ in range(75)]
    layer_three = [NeuronParams(len(layer_two), 0.1, 'sigmoid', 100) for _ in range(10)]
    

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 5000)

    min_error, iterations, best_weights_history, error_history = neural_network.train(example_data_input, example_data_output)
    print(f'Min error: {min_error} - Iterations: {iterations}')
    neural_network.dump_weights_to_file('results/weights.json')

    neural_network.load_weights_from_file('results/weights.json')
    # neural_network.print_weights()

    results = []
    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data, True)
        result = [round(res, 2) for res in result]
        print(f'Result {i}: {result} - expected: {example_data_output[i]}')
        results.append(result)

    intensity = 0.3

    test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    # for i, input_data in enumerate(test_data):
    #     noise.print_number(i, input_data)

    accuracies = []
    intensities = range(1, 50)
    intensities = [i/100 for i in intensities]

    for intensity in intensities:
        test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

        accuracy = 0
        for _ in range(100):
            correct = 0
            for i, val in enumerate(test_data):
                guess = np.argmax(neural_network.predict(val))
                if example_data_output[i][guess] == 1:
                    correct += 1

            accuracy += float(correct)/len(example_data_input)
        
        accuracy = accuracy / 100
        accuracies.append(accuracy)

    plt.figure()
    plt.plot(intensities, accuracies)
    plt.xlabel('Intensity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Intensity - Gaussian Noise')

    # accuracies = []
    # intensities = range(1, 50)
    # intensities = [i/100 for i in intensities]

    # for intensity in intensities:
    #     test_data = [noise.random_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    #     accuracy = 0
    #     for _ in range(100):
    #         correct = 0
    #         for i, val in enumerate(test_data):
    #             guess = np.argmax(neural_network.predict(val))
    #             if example_data_output[i][guess] == 1:
    #                 correct += 1

    #         accuracy += float(correct)/len(example_data_input)
        
    #     accuracy = accuracy / 100
    #     accuracies.append(accuracy)

    # plt.figure()
    # plt.plot(intensities, accuracies)
    # plt.xlabel('Intensity')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Intensity - Random Noise')

    # accuracies = []
    # intensities = range(1, 50)
    # intensities = [i/100 for i in intensities]

    # for intensity in intensities:
    #     test_data = [noise.salt_and_pepper_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    #     accuracy = 0
    #     for _ in range(100):
    #         correct = 0
    #         for i, val in enumerate(test_data):
    #             guess = np.argmax(neural_network.predict(val))
    #             if example_data_output[i][guess] == 1:
    #                 correct += 1

    #         accuracy += float(correct)/len(example_data_input)
        
    #     accuracy = accuracy / 100
    #     accuracies.append(accuracy)

    # plt.figure()
    # plt.plot(intensities, accuracies)
    # plt.xlabel('Intensity')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Intensity - Salt and Pepper Noise')
      
   
    # fig, axs = plt.subplots(len(results), figsize=(10, 30))

    #   # For each input, create a bar chart of the result and expected result
    # for i, (result, expected) in enumerate(zip(results, example_data_output)):
    #     axs[i].bar(np.arange(len(result)), result, alpha=0.7, label='Result')
    #     axs[i].bar(np.arange(len(expected)), expected, alpha=0.7, label='Expected')
    #     axs[i].set_title(f'Input {i+1}')
    #     axs[i].legend()

    plt.show()

if __name__ == "__main__":
    main()
