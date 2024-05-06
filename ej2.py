import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

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

    def compute_excitement(self, value):
        return sum(value * self.weights) + self.bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)
    
    def predict(self, data_input):
        return [self.compute_activation(self.compute_excitement(value)) for value in data_input]
    
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
        while self.min_error > 0.1 and i < max_epochs:
            mu = np.random.randint(0, len(data_input))
            value = data_input[mu]
           
            base_delta = self.calculate_delta(value, expected_output[mu])

            self.bias = self.bias + base_delta
            self.weights = self.weights + base_delta * value

            error = sum((expected_output[mu] - self.compute_activation(self.compute_excitement(data_input[mu])))**2 for mu in range(0, len(data_input)))/len(data_input)
      
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
    return sum([abs(expected_output[mu] - data[mu])**2 for mu in range(0, len(data))])/2

def main():
    data = pd.read_csv('TP3-ej2-conjunto.csv')

    example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    example_data_output = data['y'].values

    scaler = MinMaxScaler(feature_range=(-1, 1))

    example_data_output = scaler.fit_transform(example_data_output.reshape(-1, 1))
    example_data_output = example_data_output.ravel()

    training_percentage = 0.7
    cutoff = int(len(example_data_input) * training_percentage)

    training_data = example_data_input[:cutoff]
    training_output = example_data_output[:cutoff]

    test_data = example_data_input[cutoff:]
    test_output = example_data_output[cutoff:]

    min_errors = []
    test_errors = []

    # for i in range(20):
    #     min_error_acum = 0
    #     test_error_acum = 0
    #     for _ in range(2):
    #         perceptron = Perceptron(5000, len(training_data[0]), 0.01, activation_function='tan_h', beta=i/10)

    #         weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output)

    #         predictions = perceptron.predict(test_data)
    #         test_error = calculate_error(predictions, test_output)

    #         min_error_acum += min_error
    #         test_error_acum += test_error

    #     min_errors.append(min_error_acum/10)
    #     test_errors.append(test_error_acum/10)

    # plt.plot(range(20), min_errors, label='Training Error')
    # plt.plot(range(20), test_errors, label='Test Error')
    # plt.ylim(0, 1)
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.legend()

    activation_function = 'tan_h'
    beta = 0.3
    learning_rate = 0.001

    perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
    # weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output)
    # print("Weights: ", weights, "Bias: ", bias)

    # predictions = perceptron.predict(test_data)
    # test_error = calculate_error(predictions, test_output)

    # print("Test error: ", test_error)

    k = 7 
    fold_size = int(len(training_data) / k)
    start = 0
    end = fold_size
 
    fold_errors_fold = []
    fold_errors_fold = []
    for j in range(k):
        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
        test_data = training_data[start:end]
        test_output = training_output[start:end]
        training_data_fold = np.concatenate((training_data[:start], training_data[end:]))
        training_output_fold = np.concatenate((training_output[:start], training_output[end:]))

        test_errors = []
        train_errors = []
      
        for _ in range(150):
            weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data_fold, training_output_fold, 1)
            predictions = perceptron.predict(test_data)
            test_error = calculate_error(predictions, test_output)
            test_errors.append(test_error)
            train_errors.append(min_error)

        fold_errors_fold.append(test_errors[len(test_errors) - 1])
        fold_errors_fold.append(train_errors[len(train_errors) - 1])


        plt.figure()
    
        plt.plot(range(len(train_errors)), train_errors, label='Training Error')
        plt.plot(range(len(test_errors)), test_errors, label='Test Error')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Error vs Epochs Fold {j + 1}')
        plt.savefig(f'results/Error_vs_Epochs_Fold_{j + 1}.png')
    
        start = end
        end += fold_size

    plt.figure()
    plt.errorbar(range(1, 8), [np.mean(errors) for errors in fold_errors_fold[1::2]], yerr=[np.std(errors) for errors in fold_errors_fold[1::2]], fmt='o-', label='Training Error')
    plt.errorbar(range(1, 8), [np.mean(errors) for errors in fold_errors_fold[0::2]], yerr=[np.std(errors) for errors in fold_errors_fold[0::2]], fmt='o-', label='Test Error')
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title('Error vs Fold')
    plt.legend()
    plt.savefig('results/Error_vs_Fold.png')

    training_percentaje_errors = []
    test_percentage_errors = []
    for training_percentage in [i/10 for i in range(1, 10)]:
        cutoff = int(len(example_data_input) * training_percentage)
        
        # Define your lower and upper percentiles
        lower_percentile = 5
        upper_percentile = 100 - lower_percentile

        # Convert your data to numpy arrays for easier manipulation
        training_data = example_data_input[:cutoff]
        training_output = example_data_output[:cutoff]

        # Calculate the IQR of the training data
        Q1 = np.percentile(training_output, lower_percentile)
        Q3 = np.percentile(training_output, upper_percentile)
        IQR = Q3 - Q1

        # Define the outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature
        outlier_indices = np.where((training_output < Q1 - outlier_step) | (training_output > Q3 + outlier_step))
        outlier_indices = np.array(outlier_indices).flatten()

        # Separate the outliers
        training_data_outliers = training_data[outlier_indices]
        training_output_outliers = training_output[outlier_indices]

        # Remove the outliers from the original training data
        np.delete(training_data, outlier_indices)
        np.delete(training_output, outlier_indices)

        # Add the outliers to the test data
        test_data = np.concatenate((example_data_input[cutoff:], training_data_outliers))
        test_output = np.concatenate((example_data_output[cutoff:], training_output_outliers))
        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)

        test_errors = []
        train_errors = []
        for _ in range(150):
            weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 1)
            train_errors.append(min_error)
            predictions = perceptron.predict(test_data)
            test_error = calculate_error(predictions, test_output)
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