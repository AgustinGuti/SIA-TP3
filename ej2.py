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
    learning_rate = 0.1

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
            train_errors.append(min_error)
            predictions = perceptron.predict(test_data)
            test_error = calculate_error(predictions, test_output)
            test_errors.append(test_error)


        # plt.figure()
    
        # plt.plot(range(len(train_errors)), train_errors, label='Training Error')
        # plt.plot(range(len(test_errors)), test_errors, label='Test Error')
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Error')
        # plt.title(f'Error vs Epochs Fold {j + 1}')
        # plt.savefig(f'results/Error_vs_Epochs_Fold_{j + 1}.png')
    
        start = end
        end += fold_size

    training_percentage = 0.7
    cutoff = int(len(example_data_input) * training_percentage)
    
    # Define your lower and upper percentiles
    lower_percentile = 25
    upper_percentile = 75

    # Convert your data to numpy arrays for easier manipulation
    training_data = example_data_input[:cutoff]
    training_output = example_data_output[:cutoff]

    print(training_data)

    # Calculate the IQR of the training data
    Q1 = np.percentile(training_data, lower_percentile)
    Q3 = np.percentile(training_data, upper_percentile)
    IQR = Q3 - Q1

    # Define the outlier step
    outlier_step = 1.5 * IQR

    # Determine a list of indices of outliers for feature
    outlier_indices = np.where((training_data < Q1 - outlier_step) | (training_data > Q3 + outlier_step))
    outlier_indices = np.array(outlier_indices).flatten()

    # Separate the outliers
    training_data_outliers = training_data[outlier_indices]
    print('outlier indices: '   , outlier_indices)
    training_output_outliers = training_output[outlier_indices]

    # Remove the outliers from the original training data
    np.delete(training_data, outlier_indices)
    print('training data: ', training_data)
    print(f'Outliers: {training_data_outliers}')
    np.delete(training_output, outlier_indices)

    # Add the outliers to the test data
    test_data = np.concatenate((example_data_input[cutoff:], training_data_outliers))
    test_output = np.concatenate((example_data_output[cutoff:], training_output_outliers))
    print('training data: ', training_data)
    perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)

    test_errors = []
    train_errors = []
    for _ in range(150):
        weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 1)
        train_errors.append(min_error)
        predictions = perceptron.predict(test_data)
        test_error = calculate_error(predictions, test_output)
        test_errors.append(test_error)
    

    plt.figure()
    
    plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    plt.plot(range(len(test_errors)), test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs Epochs')
    plt.show()




    # df = pd.DataFrame(example_data_input, columns=[col for col in data.columns if col.startswith('x')])
    # df['output'] = example_data_output

    # X = df[[col for col in df.columns if col.startswith('x')]].values
    # y = df['output'].values

    # x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), num=10)
    # y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), num=10)

    # regressor = LinearRegression()
    # regressor.fit(X, y)
    # print('Weights: ', regressor.coef_)
    # print('Bias: ', regressor.intercept_)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    # ax2.scatter(X[:cutoff, 0], X[:cutoff, 1], y[:cutoff], color='green', label='Training Data')
    # ax2.scatter(X[cutoff:, 0], X[cutoff:, 1], y[cutoff:], color='red', label='Test Data')

    # x_values, y_values = np.meshgrid(x_range, y_range)

    # xy_matrix = np.c_[x_values.ravel(), y_values.ravel()]

    # # Predict the z values for each (x, y) pair
    # z_values = perceptron.predict(xy_matrix)

    # # Reshape the z_values array to match the shape of the x_values and y_values arrays
    # z_values = np.array(z_values).reshape(x_values.shape)

    # ax2.plot_surface(x_values, y_values, z_values, alpha=0.5, rstride=100, cstride=100, label='Regression Plane')

    # plt.legend(['Training Data', 'Test Data'])

    # plt.show()

if __name__ == "__main__":
    main()