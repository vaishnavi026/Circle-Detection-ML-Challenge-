import torch
import tqdm
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               calculate_accuracy,
               device: torch.device= "cuda"):
    train_loss, train_acc = 0, 0
    model.train()


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

    print(f"Train Loss: {train_loss: .5f} | Train Accuracy: {train_acc:.2f}")
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

        print(f"Test Loss: {test_loss: .5f} | Test Accuracy: {test_acc:.2f}")
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