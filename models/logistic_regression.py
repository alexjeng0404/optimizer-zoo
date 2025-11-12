import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression model for MNIST classification
    This is essentially a single linear layer
    """
    def __init__(self, input_size=784, output_size=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28*28)
        
        # Linear transformation
        x = self.linear(x)
        return x
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiClassLogisticRegression(nn.Module):
    """
    Alternative implementation with explicit weight and bias
    """
    def __init__(self, input_size=784, output_size=10):
        super(MultiClassLogisticRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(output_size, input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.input_size)
        
        # Linear transformation: x * W^T + b
        x = torch.matmul(x, self.weights.t()) + self.bias
        return x
    
    def get_num_parameters(self):
        return self.weights.numel() + self.bias.numel()


if __name__ == "__main__":
    # Test the model
    model = LogisticRegression()
    print(f"Logistic Regression Parameters: {model.get_num_parameters():,}")
    
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")