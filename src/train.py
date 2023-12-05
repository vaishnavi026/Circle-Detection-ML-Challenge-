import torch
import tqdm
from tqdm.auto import tqdm
import torch.nn as nn

from src.model import CNNModel
from timeit import default_timer as timer

from src.utils import print_train_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               calculate_accuracy,
               device: torch.device= "cuda"):
    train_loss, train_acc = 0, 0
    model.train()
    print(device)

    # Loop through training batches
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss # accumulate train loss
        train_acc += calculate_accuracy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader) #(average loss per batch)
    train_acc /= len(data_loader)

    print(f"Train Loss: {train_loss: .5f} | Train Accuracy: {train_acc:.2f}", flush=True)
    return (train_loss, train_acc)

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               calculate_accuracy,
               device: torch.device= "cuda"):

    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)

            test_acc += calculate_accuracy(predictions=test_pred, targets=y_test)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test Loss: {test_loss: .5f} | Test Accuracy: {test_acc:.2f}", flush=True)
        return (test_loss, test_acc)
    
def eval_model(model,
               data_loader,
               loss_fn,
               calculate_accuracy, device ="cuda"):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += calculate_accuracy(predictions=y_pred, targets=y)
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
          "model_loss": loss.item(),
          "model_acc": acc}

def train(train_loader, test_loader, calculate_accuracy, epochs = 50, device='cuda'):
    torch.manual_seed(42)
    model = CNNModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    start_time= timer()
    train_losses = []
    test_losses = []
    test_accs = []
    epochs_ = []

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n--------", flush = True)
        train_loss, train_acc = train_step(model = model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                calculate_accuracy=calculate_accuracy,
                device=device
                )

        test_loss, test_acc = test_step(model = model,
                data_loader=test_loader,
                loss_fn=loss_fn,
                calculate_accuracy=calculate_accuracy,
                device=device
                )

        epochs_.append(epoch)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        test_accs.append(test_acc)





        # Calculate training time
        end_time = timer()
        total_train_time_model = print_train_time(start=start_time,
                                                    end=end_time,
                                                    device=str(next(model.parameters()).device))