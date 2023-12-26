import torch

def accuracy(predictions: torch.Tensor, ground_truths: torch.Tensor):
    correct = torch.eq(predictions, ground_truths).sum().item()
    # correct = (predictions == ground_truths).sum().item()
    acc = correct / len(predictions) * 100
    return acc