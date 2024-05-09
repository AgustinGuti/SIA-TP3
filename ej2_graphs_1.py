from ej2 import Perceptron, Scaler, calculate_error, calculate_errors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('TP3-ej2-conjunto.csv')

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
            'beta': 0.7,
            'learning_rate': 0.1
        },
        {
            'activation_function': 'linear',
            'beta': 1,
            'learning_rate': 0.001
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

        training_data = original_example_data_input

        training_output = example_data_output

        k = 4
        fold_size = int(len(training_data) / k)
        start = 0
        end = fold_size
    
        test_errors_fold = []
        train_errors_fold = []
        for j in range(k-1):
            perceptron = Perceptron(len(training_data[0]), learning_rate, activation_function=activation_function, beta=beta)
            test_data = training_data[start:end]
            test_output = training_output[start:end]
            training_data_fold = np.concatenate((training_data[:start], training_data[end:]))
            training_output_fold = np.concatenate((training_output[:start], training_output[end:]))

            test_errors = []
            train_errors = []
            last_min_error = None
            iters_without_improvement = 0
            for _ in range(5000):
                perceptron.train(training_data_fold, training_output_fold, 1)
            
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

                if last_min_error is not None and train_error >= last_min_error:
                    iters_without_improvement += 1
                    if iters_without_improvement == 500:
                        break
                else:
                    iters_without_improvement = 0
                    last_min_error = train_error

            test_errors_fold.append(test_errors)
            train_errors_fold.append(train_errors)

            start = end
            end = end + fold_size
      
        for i in range(len(test_errors_fold)):
            if len(test_errors_fold[i]) < 3000:
                test_errors_fold[i] = test_errors_fold[i][:-475]
                train_errors_fold[i] = train_errors_fold[i][:-475]

        max_len = max([len(errors) for errors in test_errors_fold])
        for i in range(len(test_errors_fold)):
            test_errors_fold[i] = test_errors_fold[i] + [test_errors_fold[i][-1]] * (max_len - len(test_errors_fold[i]))
            train_errors_fold[i] = train_errors_fold[i] + [train_errors_fold[i][-1]] * (max_len - len(train_errors_fold[i]))

        # Sample the data
        sample_rate = max(1, len(train_errors_fold[0]) // 50)

        train_errors_fold_sampled = [errors[::sample_rate] for errors in train_errors_fold]
        test_errors_fold_sampled = [errors[::sample_rate] for errors in test_errors_fold]

        sampled_epochs = range(0, len(train_errors_fold_sampled[0]) * sample_rate, sample_rate)

        min_train_error = np.min(np.mean(train_errors_fold_sampled, axis=0))
        min_test_error = np.min(np.mean(test_errors_fold_sampled, axis=0))

        plt.figure()   

        plt.errorbar(sampled_epochs, np.mean(train_errors_fold_sampled, axis=0), yerr=np.std(train_errors_fold_sampled, axis=0), capsize=5, fmt='o-', label='Train Error', color='#1F77B4', ecolor='#1F77B4', alpha=0.5)
        plt.errorbar(sampled_epochs, np.mean(test_errors_fold_sampled, axis=0), yerr=np.std(test_errors_fold_sampled, axis=0), capsize=5, fmt='o-', label='Test Error',  color='#FF7F0E', ecolor='#FF7F0E', alpha=0.5)

        plt.text(sampled_epochs[-1], np.mean(train_errors_fold_sampled, axis=0)[-1], f'Train Error: {min_train_error:.2f}', ha='right', va='top')
        plt.text(sampled_epochs[-1], np.mean(test_errors_fold_sampled, axis=0)[-1], f'Test Error: {min_test_error:.2f}', ha='right', va='bottom')


        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Error vs Epoch - {config["activation_function"]}')
        plt.legend()
        plt.savefig(f'results/Error_vs_Epoch_graphs_1_{config["activation_function"]}.png')


    plt.show()

if __name__ == "__main__":
    main()