import os
import pickle
# import utils
from src.utils import generate_examples
from skimage.draw import circle_perimeter_aa
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def generate_data(noise_level=0.5, img_size=100, num_samples=1000):
    # Define the file path based on the configurations
    file_path = f'data/noise_{noise_level}_size_{img_size}_samples_{num_samples}.pkl'

    # Adjust the file path to the desired location outside the "src" folder
    file_path = os.path.join(os.pardir, file_path)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(file_path)

    # if os.path.exists(file_path):
    #     # If the pickle file exists, load the data
    #     with open(file_path, 'rb') as file:
    #         dataset = pickle.load(file)
    #     print(f'Data loaded from {file_path}')
    # else:
        # If the pickle file does not exist, generate the data and save it
    min_radius = img_size // 10
    max_radius = img_size // 2

    data_generator = generate_examples(noise_level=noise_level, img_size=img_size, min_radius=min_radius, max_radius=max_radius)

    # Generate and save the data
    dataset = [next(data_generator) for _ in range(num_samples)]

    # Save the dataset using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)
    print(f'Data saved to {file_path}')

    return dataset

def get_train_test_data(noise_level=0.5, img_size=100, num_samples=1000, batch_size=32):
    data = generate_data(noise_level, img_size, num_samples)
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Access images and parameters
    train_images, train_params = zip(*train_data)
    test_images, test_params = zip(*test_data)

    X_train = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(train_params, dtype=torch.float32)

    X_test = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
    Y_test = torch.tensor(test_params, dtype=torch.float32)

    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(dataset= train_data,
                            batch_size= batch_size,
                            shuffle=True)

    test_loader = DataLoader(dataset= test_data,
                            batch_size= batch_size,
                            shuffle=False)
    print(f"DataLoader: {train_loader, test_loader}")
    print(f"Length of train data loader: {len(train_loader)} batches of {batch_size}")
    print(f"Length of test data loader: {len(test_loader)} batches of {batch_size}")
    return (train_loader, test_loader)

