import os
import pickle
import utils
from src.utils import generate_examples
from skimage.draw import circle_perimeter_aa

def get_data(file_path='circle_data.pkl', noise_level=0.5, img_size=100, min_radius=None, max_radius=None, num_examples=1000):
    if os.path.exists(file_path):
        # If the pickle file exists, load the data
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        print(f'Data loaded from {file_path}')
    else:
        # If the pickle file does not exist, generate the data and save it
        min_radius = min_radius or img_size // 10
        max_radius = max_radius or img_size // 2

        data_generator = generate_examples(noise_level=noise_level, img_size=img_size, min_radius=min_radius, max_radius=max_radius)

        # Generate and save the data
        dataset = [next(data_generator) for _ in range(num_examples)]

        # Save the dataset using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)
        print(f'Data saved to {file_path}')

    return dataset

# Example usage:
data = get_data()
