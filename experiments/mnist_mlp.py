import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time

# Import from utils
from utils.data_utils import get_mnist_loaders
from utils.plot_utils import plot_optimizer_comparison, create_summary_table, plot_multiple_metrics
from utils.loss_functions import get_criterion, calculate_accuracy
from utils.save_results import save_results_to_json, save_training_plots

# Import models and optimizers
from models.mlp import MLP
from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.adam import Adam
from optimizers.adabelief import AdaBelief
from optimizers.rmsprop import Rmsprop
from optimizers.adagrad import Adagrad
from optimizers.adamw import AdamW


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=5, device='cpu'):
    """
    Train the model and return training history
    """
    model.to(device)
    train_losses = []
    test_accuracies = []
    
    print(f"Training on {device}")
    print(f"Model has {model.get_num_parameters():,} parameters")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    return train_losses, test_accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def compare_optimizers():
    """
    Compare different optimizers on MLP model
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preparation using utility function
    train_loader, _, test_loader = get_mnist_loaders(batch_size=64, data_dir='../data')
    
    # Optimizers to compare
    optimizers_config = [
        ('SGD', SGD, {'lr': 0.01}),
        ('Momentum', Momentum, {'lr': 0.01, 'beta': 0.9}),
        ('Adam', Adam, {'lr': 0.001}),
        ('AdaBelief', AdaBelief, {'lr': 0.001}),
        ('RMSProp', Rmsprop, {'lr': 0.001}),
        ('AdaGrad', Adagrad, {'lr': 0.1}),
        ('AdamW', AdamW, {'lr': 0.001, 'weight_decay': 0.01}),
    ]
    
    results = {}
    
    for opt_name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n{'='*50}")
        print(f"Testing {opt_name}")
        print(f"{'='*50}")
        
        # Create new model for each optimizer
        model = MLP().to(device)
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        criterion = get_criterion('cross_entropy')  # Using utility function
        
        # Train the model
        start_time = time.time()
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, optimizer, criterion, 
            epochs=5, device=device
        )
        training_time = time.time() - start_time
        
        results[opt_name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'training_time': training_time,
            'final_accuracy': test_accuracies[-1]
        }
        
        print(f"{opt_name} completed in {training_time:.2f} seconds")
        print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    return results


def main():
    """
    Main function to run the comparison
    """
    print("MLP Optimizer Comparison on MNIST")
    print("=" * 50)
    
    # Ensure directories exist
    os.makedirs('../results/loss_curves', exist_ok=True)
    
    results = compare_optimizers()
    
    if results:
        # save result
        save_results_to_json(results, 'results/logs/mlp_results.json')
        save_training_plots(results, 'mlp')
        
        create_summary_table(results)
        
        # plot
        plot_multiple_metrics(results)
        
        print("✅ MLP experiment completed successfully!")
        return results
    else:
        print("❌ MLP experiment failed - no results")
        return None


if __name__ == "__main__":
    main()