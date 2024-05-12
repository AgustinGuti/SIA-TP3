import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Perceptron:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100):
        self.weights = np.random.rand(dimensions)
        self.bias = np.random.rand()
        self.error = None
        self.min_error = sys.maxsize
        self.min_weights = None
        self.min_bias = None
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta

    activation_functions = {
        'linear': lambda x, b=0: x,
        'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
        'tan_h': lambda x, b: np.tanh(b*x)
    }

    derivative_activation_functions = {
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2)
    }        

    def compute_excitement(self, value, best=False):
        weights = self.weights if not best else self.min_weights
        bias = self.bias if not best else self.min_bias
        return sum(value * weights) + bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)
    
    def process(self, data_input):
        return [self.compute_activation(self.compute_excitement(value, False)) for value in data_input]

    def predict(self, data_input):
        return [self.compute_activation(self.compute_excitement(value, True)) for value in data_input]
    
    def calculate_delta(self, value, expected_output):
        excitement = self.compute_excitement(value)
        activation = self.compute_activation(excitement)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)

        return self.learning_rate * (expected_output - activation) * derivative

    def train(self, data_input, expected_output, max_epochs=1000):
        i = 0
        weight_history = []
        error_history = []
        bias_history = []
        weight_history.append(self.weights)
        error_history.append(self.min_error)
        bias_history.append(self.bias)
        while self.min_error > 0 and i < max_epochs:
            mu = np.random.randint(0, len(data_input))
            value = data_input[mu]
           
            base_delta = self.calculate_delta(value, expected_output[mu])

            self.bias = self.bias + base_delta
            self.weights = self.weights + base_delta * value

            predictions = self.process(data_input)
            error = calculate_error(predictions, expected_output)
      
            if error < self.min_error:
                weight_history.append(self.weights)
                bias_history.append(self.bias)
                error_history.append(error)
                self.min_error = error
                self.min_weights = self.weights
                self.min_bias = self.bias
            i += 1

        # print("Iterations: ", i)
        # print("Error:", self.min_error)
        return self.min_weights, self.min_bias, self.min_error, weight_history, error_history, bias_history

def calculate_error(data, expected_output):
    return sum(calculate_errors(data, expected_output))/len(data)

def calculate_errors(data, expected_output):
    return [abs((expected_output[mu] - data[mu])**2) for mu in range(0, len(data))]

class Scaler:
    def __init__(self, min_val, max_val, activation_function='linear'):
        self.min_val = min_val
        self.max_val = max_val
        self.activation_function = activation_function

    def fit_transform(self, data):
        if self.activation_function == 'linear':
            return data
        elif self.activation_function == 'tan_h':
            return (data - self.min_val) / (self.max_val - self.min_val) * 2 - 1
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data):
        if self.activation_function == 'linear':
            return data
        elif self.activation_function == 'tan_h':
            return ((np.array(data) + 1) / 2 * (self.max_val - self.min_val)) + self.min_val
        return (np.array(data) * (self.max_val - self.min_val)) + self.min_val

def main():
    data = pd.read_csv('TP3-ej2-conjunto.csv')

    example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    example_data_output = data['y'].values

    perm = np.random.permutation(len(example_data_input))

    example_data_input = example_data_input[perm]
    example_data_output = example_data_output[perm]

    activation_function = 'tan_h'
    beta = 0.5
    learning_rate = 0.05

    # Min-Max scaling
    min_val = np.min(example_data_output)
    max_val = np.max(example_data_output)

    scaler = Scaler(min_val, max_val, activation_function)
    example_data_output = scaler.fit_transform(example_data_output)

    training_percentage = 0.8
    cutoff = int(len(example_data_input) * training_percentage)

    training_data = example_data_input[:cutoff]
    training_output = example_data_output[:cutoff]

    test_data = example_data_input[cutoff:]
    test_output = example_data_output[cutoff:]

    min_errors = []
    test_errors = []

    perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)


    k = 7
    fold_size = int(len(training_data) / k)
    start = 0
    end = fold_size
 
    fold_errors_fold = []
    test_errors_fold = []
    for j in range(k-1):
        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
        test_data = training_data[start:end]
        test_output = training_output[start:end]
        training_data_fold = np.concatenate((training_data[:start], training_data[end:]))
        training_output_fold = np.concatenate((training_output[:start], training_output[end:]))

        test_errors = []
        train_errors = []
      
        for _ in range(3000):
            weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data_fold, training_output_fold, 1)

            train_predictions = perceptron.predict(training_data_fold)
            local_train_predictions = scaler.inverse_transform(train_predictions)
            local_output_fold = scaler.inverse_transform(training_output_fold)
            train_error = calculate_error(local_train_predictions, local_output_fold)

            predictions = perceptron.predict(test_data)
            local_predictions = scaler.inverse_transform(predictions)
            local_test_output = scaler.inverse_transform(test_output)
            test_error = calculate_error(local_predictions, local_test_output)

            test_errors.append(test_error)
            train_errors.append(train_error)

        test_errors_fold.append(test_errors[len(test_errors) - 1])
        fold_errors_fold.append(train_errors[len(train_errors) - 1])


        plt.figure()
    
        plt.plot(range(len(train_errors)), train_errors, label='Training Error')
        plt.plot(range(len(test_errors)), test_errors, label='Test Error')
        plt.legend()
        plt.ylim(0, 300)
        plt.xlabel('Epochs')
        plt.ylabel('Error')

        plt.text(len(train_errors), train_errors[-1], f'Train Error: {train_errors[-1]:.2f}', ha='right', va='top')
        plt.text(len(test_errors), test_errors[-1], f'Test Error: {test_errors[-1]:.2f}', ha='right', va='bottom')

        # plt.text(sampled_epochs[-1], np.mean(test_errors_fold_sampled, axis=0)[-1], f'Test Error: {min_test_error:.2f}', ha='right', va='bottom')
        plt.title(f'Error vs Epochs Fold {j + 1}')
        plt.savefig(f'results/Error_vs_Epochs_Fold_{j + 1}.png')
    
        start = end
        end += fold_size

    plt.figure()
    plt.errorbar(range(1, k), fold_errors_fold, fmt='o-', label='Training Error')
    plt.errorbar(range(1, k), test_errors_fold, fmt='o-', label='Test Error')
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title('Error vs Fold')
    plt.legend()
    plt.savefig('results/Error_vs_Fold.png')

    training_percentaje_errors = []
    test_percentage_errors = []
    for training_percentage in [i/10 for i in range(1, 10)]:
        cutoff = int(len(example_data_input) * training_percentage)

        # Convert your data to numpy arrays for easier manipulation
        training_data = example_data_input[:cutoff]
        training_output = example_data_output[:cutoff]

        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)

        test_errors = []
        train_errors = []
        for _ in range(5000):
            weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 1)
            
            train_predictions = perceptron.predict(training_data)
            train_predictions = scaler.inverse_transform(train_predictions)
            local_training_output = scaler.inverse_transform(training_output)
            train_error = calculate_error(train_predictions, local_training_output)
            train_errors.append(train_error)


            predictions = perceptron.predict(test_data)
            predictions = scaler.inverse_transform(predictions)
            local_test_output = scaler.inverse_transform(test_output)
            test_error = calculate_error(predictions, local_test_output)
            test_errors.append(test_error)
    
        training_percentaje_errors.append(train_errors[len(train_errors) - 1])
        test_percentage_errors.append(test_errors[len(test_errors) - 1])

        # plt.figure()
        
        # plt.plot(range(len(train_errors)), train_errors, label='Training Error')
        # plt.plot(range(len(test_errors)), test_errors, label='Test Error')
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Error')
        # plt.title(f'Error vs Epochs - Training Percentage {training_percentage}')

    plt.figure()
    plt.errorbar([i/10 for i in range(1, 10)], training_percentaje_errors, fmt='o-', label='Training Error')
    plt.errorbar([i/10 for i in range(1, 10)], test_percentage_errors, fmt='o-', label='Test Error')
    plt.xlabel('Training Percentage')
    plt.ylabel('Error')
    plt.title('Error vs Training Percentage')
    plt.legend()
    plt.savefig('results/Error_vs_Training_Percentage.png')


    plt.show()

if __name__ == "__main__":
    main()