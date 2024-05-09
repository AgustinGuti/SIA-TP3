from ej2 import Perceptron, Scaler, calculate_error, calculate_errors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('TP3-ej2-conjunto copy.csv')

    original_example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    original_example_data_output = data['y'].values
   

    configs = [
        {
            'activation_function': 'tan_h',
            'beta': 1.3,
            'learning_rate': 0.1
        },        
        {
            'activation_function': 'sigmoid',
            'beta': 3.9,
            'learning_rate': 0.1
        },
        {
            'activation_function': 'linear',
            'beta': 1,
            'learning_rate': 0.01
        }
    ]

    for config in configs:
    
        activation_function = config['activation_function']
        beta = config['beta']
        learning_rate = config['learning_rate']


        # Min-Max scaling
        min_val = np.min(original_example_data_output)
        max_val = np.max(original_example_data_output)

        scaler = Scaler(min_val, max_val, activation_function)
        example_data_output = scaler.fit_transform(original_example_data_output)

        # perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)

        test_percentage = 0.2
        cutoff = int(len(original_example_data_input) * (1 - test_percentage))

        training_data = original_example_data_input[:cutoff]
        training_output = example_data_output[:cutoff]

        test_data = original_example_data_input[cutoff:]
        test_output = example_data_output[cutoff:]

        perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
        weights, bias, min_error, weight_history, error_history, bias_history = perceptron.train(training_data, training_output, 1000)



        x = np.linspace(min(training_data[:, 0]), max(training_data[:, 0]), 100)
        y = np.linspace(min(training_data[:, 1]), max(training_data[:, 1]), 100)
        X, Y = np.meshgrid(x, y)

        Z = np.array(perceptron.predict(np.array([X.flatten(), Y.flatten()]).T)).reshape(100, 100)
        Z = scaler.inverse_transform(Z)
        local_train_output = scaler.inverse_transform(training_output)
        local_test_output = scaler.inverse_transform(test_output)

        z_max = max([max(local_train_output), max(local_test_output)])
        z_min = min([min(local_train_output), min(local_test_output)])

        local_train_predictions = scaler.inverse_transform(perceptron.predict(training_data))
        train_output_predictions_error = np.array(calculate_errors(local_train_predictions, local_train_output))
        
        local_test_predictions = scaler.inverse_transform(perceptron.predict(test_data))

        test_output_predictions_error = np.array(calculate_errors(local_test_predictions, local_test_output))



        # Normalize errors
       

        train_output_predictions_error_scaled = (train_output_predictions_error - min(train_output_predictions_error)) / (max(train_output_predictions_error) - min(train_output_predictions_error)) + 1
        test_output_predictions_error_scaled = (test_output_predictions_error - min(test_output_predictions_error)) / (max(test_output_predictions_error) - min(test_output_predictions_error)) + 1

        global_error_scaled_min = min(min(train_output_predictions_error_scaled), min(test_output_predictions_error_scaled))
        global_error_scaled_max = max(max(train_output_predictions_error_scaled), max(test_output_predictions_error_scaled))

        train_output_predictions_error_scaled = ((train_output_predictions_error - global_error_scaled_min) / (global_error_scaled_max - global_error_scaled_min)) 
        test_output_predictions_error_scaled = ((test_output_predictions_error - global_error_scaled_min) / (global_error_scaled_max - global_error_scaled_min)) 

        global_error_min = min(min(train_output_predictions_error), min(test_output_predictions_error))
        global_error_max = max(max(train_output_predictions_error), max(test_output_predictions_error))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')

        scatter1 = ax.scatter(test_data[:, 0], test_data[:, 1], local_test_output, c=test_output_predictions_error, label='Test Data', cmap='RdYlGn_r', s=test_output_predictions_error_scaled, vmin=global_error_min, vmax=global_error_max)
        plt.colorbar(scatter1, ax=ax)
        ax.legend()
        ax.plot_surface(X, Y, Z)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        avg_test_error = sum(test_output_predictions_error)/len(test_output_predictions_error)

        plt.title(f'Perceptron Surface - {config["activation_function"]} - Test Data - Error: {avg_test_error:.2f}')
        plt.savefig(f'results/Surface_graphs_1_test_{config["activation_function"]}.png')

        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(training_data[:, 0], training_data[:, 1], local_train_output, c=train_output_predictions_error, label='Training Data', cmap='RdYlGn_r', s=train_output_predictions_error_scaled, vmin=global_error_min, vmax=global_error_max)

        plt.colorbar(scatter2, ax=ax2)
        plt.legend()
        ax2.plot_surface(X, Y, Z)
        ax2.set_zlim(z_min, z_max)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        avg_train_error = sum(train_output_predictions_error)/len(train_output_predictions_error)

        plt.tight_layout()
        plt.title(f'Perceptron Surface - {config["activation_function"]} - Training Data - Error: {avg_train_error:.2f}')
        plt.savefig(f'results/Surface_graphs_1_train_{config["activation_function"]}.png')




    plt.show()

if __name__ == "__main__":
    main()