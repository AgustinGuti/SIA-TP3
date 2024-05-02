import numpy as np

import matplotlib.pyplot as plt


def salt_and_pepper_noise(vector, intensity=0.1):
    noise_matrix = np.array(vector).reshape(7,5)

    for i in range(7):
        for j in range(5):
            if np.random.random() < intensity:
                noise_matrix[i][j] = 1 - noise_matrix[i][j]

    return noise_matrix.flatten()

def gaussian_noise(vector, intensity=0.1):
    noise_matrix = np.array(vector).reshape(7,5)

    noise_matrix = noise_matrix + np.random.normal(0, intensity, (7,5))

    noise_matrix = noise_matrix - np.min(noise_matrix)
    noise_matrix = noise_matrix / np.max(noise_matrix)

    return noise_matrix.flatten()

def random_noise(vector, intensity=0.1):
    noise_matrix = np.array(vector, dtype=np.float64).reshape(7,5)

    noise_matrix += np.random.random((7,5)) * intensity

    return noise_matrix.flatten()
    

def print_number(number, data):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(7,5)

    # Display the data as an image
    plt.figure()
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    # plt.savefig(f'plot{number}.png', dpi=300, bbox_inches='tight')  # Adjust dpi and bbox as needed
    # plt.show()







