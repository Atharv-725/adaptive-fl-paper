import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test

def iid_split(dataset, num_clients):
    size = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    client_data = {}
    for i in range(num_clients):
        client_indices = indices[i * size : (i + 1) * size]
        client_data[i] = Subset(dataset, client_indices)
    return client_data

def apply_drift(dataset, intensity=0.5):
    data   = dataset.dataset.data[dataset.indices]
    labels = dataset.dataset.targets[dataset.indices].clone()
    indices = list(range(len(labels)))
    n_corrupt = int(len(indices) * intensity)
    corrupt_idx = np.random.choice(indices, n_corrupt, replace=False)
    for idx in corrupt_idx:
        labels[idx] = torch.randint(0, 10, (1,)).item()
    from torch.utils.data import TensorDataset
    data_float = data.float().unsqueeze(1) / 255.0
    norm = transforms.Normalize((0.1307,), (0.3081,))
    data_norm = torch.stack([norm(img) for img in data_float])
    return TensorDataset(data_norm, labels)

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
