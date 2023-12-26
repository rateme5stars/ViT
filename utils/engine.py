import torch
from torch import nn
from utils.metric import accuracy
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm.auto import tqdm

def train_step(model: nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    model.train()
    train_loss, train_acc = 0, 0

    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        batch_loss = loss_fn(y_logits, y)
        train_loss += batch_loss

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        batch_acc = accuracy(y_pred_class, y)
        train_acc += batch_acc

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(model: nn.Module, 
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module, 
              device):
    
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for _, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            batch_loss = loss_fn(y_logits, y)
            test_loss += batch_loss

            y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            batch_acc = accuracy(y_pred_class, y)
            test_acc += batch_acc

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc

def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          epochs: int = 10,) -> Dict[str, List]:
    
    results = defaultdict(list)
    model.to(device)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          data_loader=train_loader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_loader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.2f} | "
          f"train_acc: {train_acc:.2f} | "
          f"test_loss: {test_loss:.2f} | "
          f"test_acc: {test_acc:.2f}"
        )

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results