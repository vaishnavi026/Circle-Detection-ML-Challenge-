import torch
import torch.nn as nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from prepare_data import get_train_test_data
from model import CNNModel
from utils import calculate_accuracy, print_train_time
from train import *

def run(noise_level, img_size, num_samples, epochs, device='cuda'):
    # Get train and test loaders
    train_loader, test_loader = get_train_test_data(noise_level=noise_level, img_size=img_size, num_samples=num_samples)

    # Initialize the model
    model = CNNModel().to(device)

    # Set up loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Start the timer
    start_time = timer()
    print("Training started.")

    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    test_accs = []
    epochs_ = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n--------", flush=True)

        # Train the model
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            calculate_accuracy=calculate_accuracy,
            device=device
        )

        # Test the model
        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            calculate_accuracy=calculate_accuracy,
            device=device
        )

        # Append results to lists
        epochs_.append(epoch)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        test_accs.append(test_acc)

    # Calculate training time
    end_time = timer()
    total_train_time_model = print_train_time(
        start=start_time,
        end=end_time,
        device=str(next(model.parameters()).device)
    )

    print("Training completed.")

    # Save the trained model
    model_filename = f'model_noise_{noise_level}_size_{img_size}_samples_{num_samples}.pkl'
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved to {model_filename}')

    # Return results or save to a log file, etc.
    return {
        'epochs': epochs_,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'training_time': total_train_time_model
    }
