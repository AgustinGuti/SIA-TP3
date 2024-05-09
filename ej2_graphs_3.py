from ej2 import Perceptron, Scaler, calculate_error, calculate_errors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('TP3-ej2-conjunto.csv')

    original_example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    original_example_data_output = data['y'].values

    indices = np.arange(len(original_example_data_input))
    np.random.shuffle(indices)

    original_example_data_input = original_example_data_input[indices]
    original_example_data_output = original_example_data_output[indices]

    configs = [
        {
            'activation_function': 'tan_h',
            'beta': 1.3,
            'learning_rate': 0.1
        },        
        {
            'activation_function': 'sigmoid',
            'beta': 0.7,
            'learning_rate': 0.1
        },
        {
            'activation_function': 'linear',
            'beta': 1,
            'learning_rate': 0.001
        }
    ]

    v_min = 0
    v_max = 90

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = original_example_data_input[:, 0]
    y = original_example_data_input[:, 1]
    z = original_example_data_input[:, 2]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    scatter = ax.scatter(x, y, z, c = original_example_data_output, s=100, vmin=v_min, vmax=v_max)
    plt.colorbar(scatter)

    plt.title('Original Data')

    for config in configs:
    
        activation_function = config['activation_function']
        beta = config['beta']
        learning_rate = config['learning_rate']


        # Min-Max scaling
        min_val = np.min(original_example_data_output)
        max_val = np.max(original_example_data_output)

        scaler = Scaler(min_val, max_val, activation_function)
        example_data_output = scaler.fit_transform(original_example_data_output)

        test_percentage = 0.4
        cutoff = int(len(original_example_data_input) * (1 - test_percentage))

        training_data = original_example_data_input[:cutoff]
        training_output = example_data_output[:cutoff]

        test_data = original_example_data_input[cutoff:]
        test_output = example_data_output[cutoff:]

        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
        weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 2000)

        # ---------------------------------------------------
        # Trained network predictions

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x1_values = np.linspace(x_min, x_max, 15)
        x2_values = np.linspace(y_min, y_max, 15)
        x3_values = np.linspace(z_min, z_max, 15)

        x1, x2, x3 = np.meshgrid(x1_values, x2_values, x3_values)

        x1 = x1.flatten()
        x2 = x2.flatten()
        x3 = x3.flatten()

        predictions = np.array(perceptron.predict(np.array([x1, x2, x3]).T))
        predictions = scaler.inverse_transform(predictions)

        scatter = ax.scatter(x1, x2, x3, s=100, c=predictions, vmin=v_min, vmax=v_max)
        plt.colorbar(scatter)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.title(f'{config["activation_function"]} Activation Function')

        # ---------------------------------------------------
        # Errors
        training_data_predictions = perceptron.predict(training_data)
        training_data_predictions = scaler.inverse_transform(training_data_predictions)
        training_output = scaler.inverse_transform(training_output)
        training_errors = calculate_errors(training_data_predictions, training_output)

        test_data_predictions = perceptron.predict(test_data)
        test_data_predictions = scaler.inverse_transform(test_data_predictions)
        test_output = scaler.inverse_transform(test_output)
        test_errors = calculate_errors(test_data_predictions, test_output)

        min_error, max_error = np.min(training_errors), np.max(training_errors)

        # Test errors

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')

        x = test_data[:, 0]
        y = test_data[:, 1]
        z = test_data[:, 2]
        scatter = ax.scatter(x, y, z, s=100, c=test_errors, cmap='RdYlGn_r', vmin=min_error, vmax=max_error)
        plt.colorbar(scatter)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        plt.title(f'{config["activation_function"]} Activation Function Test Errors - MSE: {np.mean(test_errors):.2f}')

        # ---------------------------------------------------
        # Training errors

        ax2 = fig.add_subplot(122, projection='3d')

        x = training_data[:, 0]
        y = training_data[:, 1]
        z = training_data[:, 2]
        scatter = ax2.scatter(x, y, z, s=100, c=training_errors, cmap='RdYlGn_r', vmin=min_error, vmax=max_error)
        plt.colorbar(scatter)

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('x3')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_zlim(z_min, z_max)

        plt.tight_layout()
        plt.title(f'{config["activation_function"]} Activation Function Training Errors - MSE: {np.mean(training_errors):.2f}')


    plt.show()

if __name__ == "__main__":
    main()