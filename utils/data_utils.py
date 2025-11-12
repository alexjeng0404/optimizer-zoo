import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

def get_mnist_loaders(batch_size=64, data_dir='../data', download=True, val_split=0.1):
    """
    Get MNIST data loaders for training, validation and testing
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load MNIST data
        download: Whether to download MNIST dataset
        val_split: Fraction of training data to use for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training and test datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)
    
    # Split training data into train and validation
    if val_split > 0:
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(batch_size=64, data_dir='../data', download=True, val_split=0.1):
    """
    Get CIFAR-10 data loaders for training, validation and testing
    """
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test)
    
    # Split training data
    if val_split > 0:
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    print(f"CIFAR-10 Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"CIFAR-10 Validation samples: {len(val_dataset)}")
    print(f"CIFAR-10 Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_data_loaders(dataset_name='mnist', **kwargs):
    """
    Generic function to get data loaders for different datasets
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
        **kwargs: Additional arguments for specific dataset loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name.lower() == 'mnist':
        return get_mnist_loaders(**kwargs)
    elif dataset_name.lower() == 'cifar10':
        return get_cifar10_loaders(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {path}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss