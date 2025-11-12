import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST classification
    Architecture: 784 -> 256 -> 128 -> 10
    """
    def __init__(self, input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28*28)
        
        # First hidden layer with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation and dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation here, CrossEntropyLoss includes SoftMax)
        x = self.fc3(x)
        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ImprovedMLP(nn.Module):
    """
    Enhanced MLP with batch normalization and more flexible architecture
    """
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10, 
                 dropout_rate=0.3, use_batchnorm=True):
        super(ImprovedMLP, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.use_batchnorm = use_batchnorm
        
        # Create layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.network(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = MLP()
    print(f"MLP Parameters: {model.get_num_parameters():,}")
    
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test improved model
    improved_model = ImprovedMLP(hidden_sizes=[512, 256, 128])
    print(f"Improved MLP Parameters: {improved_model.get_num_parameters():,}")