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
        while self.min_error > 0.1 and i < max_epochs:
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
    return [abs(expected_output[mu] - data[mu])**2 for mu in range(0, len(data))]

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
    learning_rate = 0.1

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


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sizes = (test_output - test_output.min()) / (test_output.max() - test_output.min()) * 100 + 10

    x = test_data[:, 0]
    y = test_data[:, 1]
    z = test_data[:, 2]
    ax.scatter(x, y, z, s=sizes, c = test_output)

    perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
    perceptron.train(training_data, training_output, 300)

    predictions = np.array(perceptron.predict(test_data))
    test_errors = calculate_errors(predictions, test_output)

    print(f'test data: {test_data}')
    print(f'prediction: {predictions}')
    print(f'output: {test_output}')
    print(f'error: {test_errors}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = test_data[:, 0]
    y = test_data[:, 1]
    z = test_data[:, 2]
    ax.scatter(x, y, z, s=[100]*len(test_errors),  c=test_errors, cmap='RdYlGn_r')

    plt.title('Test Data Errors')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sizes = (example_data_output - example_data_output.min()) / (example_data_output.max() - example_data_output.min()) * 100 + 10

    x = example_data_input[:, 0]
    y = example_data_input[:, 1]
    z = example_data_input[:, 2]
    ax.scatter(x, y, z, s=sizes, c = example_data_output)

    plt.title('Original Data')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    predictions_1 = np.array(perceptron.predict(training_data))
    predictions_1 = scaler.inverse_transform(predictions_1)

    sizes = (predictions_1 - predictions_1.min()) / (predictions_1.max() - predictions_1.min()) * 100 + 10

    x1 = training_data[:, 0]
    y1 = training_data[:, 1]
    z1 = training_data[:, 2]
    ax.scatter(x1, y1, z1, s=[100]*len(predictions_1), c=predictions_1, label='Training Data')

    x11 = test_data[:, 0]
    y11 = test_data[:, 1]
    z11 = test_data[:, 2]

    predictions_11 = np.array(perceptron.predict(test_data))
    predictions_11 = scaler.inverse_transform(predictions_11)

    sizes = (predictions_11 - predictions_11.min()) / (predictions_11.max() - predictions_11.min()) * 100 + 10
    ax.scatter(x11, y11, z11, s=[50]*len(predictions_11), c=predictions_11, label='Test Data')

    plt.legend()
    plt.title('Predictions for Training and Test Data')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    errors1 = calculate_errors(predictions_1, training_output)
    errors2 = calculate_errors(predictions_11, test_output)
    ax.scatter(x1, y1, z1, s=[100]*len(errors1),  c=errors1, cmap='RdYlGn_r', label='Training Data')
    ax.scatter(x11, y11, z11, s=[50]*len(errors2), c=errors2, cmap='RdYlGn_r', label='Test Data')

    train_error = calculate_error(predictions_1, training_output)
    test_error = calculate_error(predictions_11, test_output)

    plt.legend()
    plt.title(f'Errors for Training and Test Data - Train Error: {train_error} - Test Error: {test_error}')




    perceptron_2 = Perceptron(len(training_data[0]), 0.1, activation_function='tan_h', beta=0.4)
    perceptron_2.train(training_data, training_output, 1000)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1_values = np.linspace(np.min(training_data[:, 0]), np.max(training_data[:, 0]), 15)
    x2_values = np.linspace(np.min(training_data[:, 1]), np.max(training_data[:, 1]), 15)
    x3_values = np.linspace(np.min(training_data[:, 2]), np.max(training_data[:, 2]), 15)

    x1, x2, x3 = np.meshgrid(x1_values, x2_values, x3_values)

    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()

    predictions_2 = np.array(perceptron_2.predict(np.array([x1, x2, x3]).T))
    predictions_2 = scaler.inverse_transform(predictions_2)

    sizes = [100]*len(predictions_2)

    ax.scatter(x1, x2, x3, s=sizes, c=predictions_2)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('TanH Activation Function')

    # perceptron_2 = Perceptron(len(training_data[0]), 0.1, activation_function='sigmoid', beta=0.3)
    # perceptron_2.train(training_data, training_output, 1000)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x1_values = np.linspace(np.min(test_data[:, 0]), np.max(test_data[:, 0]), 15)
    # x2_values = np.linspace(np.min(test_data[:, 1]), np.max(test_data[:, 1]), 15)
    # x3_values = np.linspace(np.min(test_data[:, 2]), np.max(test_data[:, 2]), 15)

    # x1, x2, x3 = np.meshgrid(x1_values, x2_values, x3_values)

    # x1 = x1.flatten()
    # x2 = x2.flatten()
    # x3 = x3.flatten()

    # predictions_2 = np.array(perceptron_2.predict(np.array([x1, x2, x3]).T))
    # predictions_2 = scaler.inverse_transform(predictions_2)

    # sizes = [100]*len(predictions_2)

    # ax.scatter(x1, x2, x3, s=sizes, c=predictions_2)

    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('x3')
    # plt.title('Sigmoid Activation Function')

    # perceptron_2 = Perceptron(len(training_data[0]), 0.001, activation_function='linear', beta=0.3)
    # perceptron_2.train(training_data, training_output, 1000)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x1_values = np.linspace(np.min(test_data[:, 0]), np.max(test_data[:, 0]), 15)
    # x2_values = np.linspace(np.min(test_data[:, 1]), np.max(test_data[:, 1]), 15)
    # x3_values = np.linspace(np.min(test_data[:, 2]), np.max(test_data[:, 2]), 15)

    # x1, x2, x3 = np.meshgrid(x1_values, x2_values, x3_values)

    # x1 = x1.flatten()
    # x2 = x2.flatten()
    # x3 = x3.flatten()

    # predictions_2 = np.array(perceptron_2.predict(np.array([x1, x2, x3]).T))
    # predictions_2 = scaler.inverse_transform(predictions_2)

    # # sizes = (predictions_2 - predictions_2.min()) / (predictions_2.max() - predictions_2.min()) * 100 + 10
    # sizes = [100]*len(predictions_2)

    # ax.scatter(x1, x2, x3, s=sizes, c=predictions_2)

    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('x3')
    # plt.title('Linear Activation Function')



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sizes = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 100 + 10

    x = test_data[:, 0]
    y = test_data[:, 1]
    z = test_data[:, 2]
    ax.scatter(x, y, z, s=sizes, c=predictions)

    x = training_data[:, 0]
    y = training_data[:, 1]
    z = training_data[:, 2]

    sizes = (training_output - training_output.min()) / (training_output.max() - training_output.min()) * 100 + 10
    # ax.scatter(x, y, z, s=sizes, c='r')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')


    # perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)


    # k = 7
    # fold_size = int(len(training_data) / k)
    # start = 0
    # end = fold_size
 
    # fold_errors_fold = []
    # fold_errors_fold = []
    # for j in range(k-1):
    #     perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
    #     test_data = training_data[start:end]
    #     test_output = training_output[start:end]
    #     training_data_fold = np.concatenate((training_data[:start], training_data[end:]))
    #     training_output_fold = np.concatenate((training_output[:start], training_output[end:]))

    #     test_errors = []
    #     train_errors = []
      
    #     for _ in range(300):
    #         weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data_fold, training_output_fold, 1)

    #         train_predictions = perceptron.predict(training_data_fold)
    #         local_train_predictions = scaler.inverse_transform(train_predictions)
    #         local_output_fold = scaler.inverse_transform(training_output_fold)
    #         train_error = calculate_error(local_train_predictions, local_output_fold)

    #         predictions = perceptron.predict(test_data)
    #         local_predictions = scaler.inverse_transform(predictions)
    #         local_test_output = scaler.inverse_transform(test_output)
    #         test_error = calculate_error(local_predictions, local_test_output)

    #         test_errors.append(test_error)
    #         train_errors.append(train_error)

    #     fold_errors_fold.append(test_errors[len(test_errors) - 1])
    #     fold_errors_fold.append(train_errors[len(train_errors) - 1])


    #     plt.figure()
    
    #     plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    #     plt.plot(range(len(test_errors)), test_errors, label='Test Error')
    #     plt.legend()
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Error')
    #     plt.title(f'Error vs Epochs Fold {j + 1}')
    #     plt.savefig(f'results/Error_vs_Epochs_Fold_{j + 1}.png')
    
    #     start = end
    #     end += fold_size

    # plt.figure()
    # plt.errorbar(range(1, k), [np.mean(errors) for errors in fold_errors_fold[1::2]], yerr=[np.std(errors) for errors in fold_errors_fold[1::2]], fmt='o-', label='Training Error')
    # plt.errorbar(range(1, k), [np.mean(errors) for errors in fold_errors_fold[0::2]], yerr=[np.std(errors) for errors in fold_errors_fold[0::2]], fmt='o-', label='Test Error')
    # plt.xlabel('Fold')
    # plt.ylabel('Error')
    # plt.title('Error vs Fold')
    # plt.legend()
    # plt.savefig('results/Error_vs_Fold.png')

    # training_percentaje_errors = []
    # test_percentage_errors = []
    # for training_percentage in [i/10 for i in range(1, 10)]:
    #     cutoff = int(len(example_data_input) * training_percentage)

    #     # Convert your data to numpy arrays for easier manipulation
    #     training_data = example_data_input[:cutoff]
    #     training_output = example_data_output[:cutoff]

    #     perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)

    #     test_errors = []
    #     train_errors = []
    #     for _ in range(150):
    #         weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 1)
    #         train_errors.append(min_error)
    #         predictions = perceptron.predict(test_data)
    #         test_error = calculate_error(predictions, test_output)
    #         test_error = scaler.inverse_transform(test_error)
    #         test_errors.append(test_error)
    
    #     training_percentaje_errors.append(train_errors[len(train_errors) - 1])
    #     test_percentage_errors.append(test_errors[len(test_errors) - 1])
    # # plt.figure()
    
    # # plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    # # plt.plot(range(len(test_errors)), test_errors, label='Test Error')
    # # plt.legend()
    # # plt.xlabel('Epochs')
    # # plt.ylabel('Error')
    # # plt.title(f'Error vs Epochs - Training Percentage {training_percentage}')

    # plt.figure()
    # plt.errorbar([i/10 for i in range(1, 10)], training_percentaje_errors, fmt='o-', label='Training Error')
    # plt.errorbar([i/10 for i in range(1, 10)], test_percentage_errors, fmt='o-', label='Test Error')
    # plt.xlabel('Training Percentage')
    # plt.ylabel('Error')
    # plt.title('Error vs Training Percentage')
    # plt.legend()
    # plt.savefig('results/Error_vs_Training_Percentage.png')


    plt.show()

if __name__ == "__main__":
    main()