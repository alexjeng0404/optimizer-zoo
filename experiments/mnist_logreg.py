import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time

# Import from utils
from utils.data_utils import get_mnist_loaders
from utils.plot_utils import plot_optimizer_comparison, create_summary_table
from utils.loss_functions import get_criterion
from utils.save_results import save_results_to_json, save_training_plots

# Import models and optimizers
from models.logistic_regression import LogisticRegression
from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.adam import Adam
from optimizers.adabelief import AdaBelief
from optimizers.rmsprop import Rmsprop
from optimizers.adagrad import Adagrad


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


def train_single_optimizer(optim_class, opt_name, opt_kwargs, train_loader, test_loader, device, epochs=10):
    """
    Train logistic regression with a single optimizer
    """
    print(f"\n{'='*50}")
    print(f"Testing {opt_name} on Logistic Regression")
    print(f"{'='*50}")
    
    # Create new model for each optimizer
    model = LogisticRegression().to(device)
    optimizer = optim_class(model.parameters(), **opt_kwargs)
    criterion = get_criterion('cross_entropy')
    
    train_losses = []
    test_accuracies = []
    
    print(f"Model has {model.get_num_parameters():,} parameters")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 2 == 0:  # Print every 2 epochs
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    training_time = time.time() - start_time
    
    print(f"{opt_name} completed in {training_time:.2f} seconds")
    print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'training_time': training_time,
        'final_accuracy': test_accuracies[-1]
    }


def run_logistic_regression_experiment():
    """
    Run logistic regression comparison experiment
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
        ('AdaGrad', Adagrad, {'lr': 0.1}),  # Higher LR for AdaGrad
    ]
    
    results = {}
    
    for opt_name, opt_class, opt_kwargs in optimizers_config:
        result = train_single_optimizer(
            opt_class, opt_name, opt_kwargs, 
            train_loader, test_loader, device, epochs=10
        )
        results[opt_name] = result
    
    return results


def main():
    """
    Main function to run logistic regression comparison
    """
    print("Logistic Regression Optimizer Comparison on MNIST")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs('../results/loss_curves', exist_ok=True)
    
    results = run_logistic_regression_experiment()
    
    if results:
        # save result
        save_results_to_json(results, 'results/logs/logreg_results.json')
        
        # save diagram
        save_training_plots(results, 'logistic_regression')
        
        # create table
        create_summary_table(results)
        
        print("✅ Logistic Regression experiment completed successfully!")
        return results
    else:
        print("❌ No results to display")
        return None


if __name__ == "__main__":
    main()