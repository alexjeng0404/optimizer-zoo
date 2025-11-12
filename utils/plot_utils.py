import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_loss_curves(train_losses, val_losses=None, title="Training Loss", save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved: {save_path}")
    
    plt.show()


def plot_accuracy_curves(train_accuracies, val_accuracies=None, test_accuracies=None, title="Accuracy", save_path=None):
    """
    Plot accuracy curves
    
    Args:
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies (optional)
        test_accuracies: List of test accuracies (optional)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    
    if val_accuracies is not None:
        plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    
    if test_accuracies is not None:
        plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved: {save_path}")
    
    plt.show()


def plot_optimizer_comparison(results, metric='test_accuracies', title="Optimizer Comparison", save_path=None):
    """
    Plot comparison of multiple optimizers
    
    Args:
        results: Dictionary with optimizer names as keys and result dictionaries as values
        metric: Metric to plot ('test_accuracies', 'train_losses', etc.)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    for opt_name, result in results.items():
        if metric in result:
            data = result[metric]
            epochs = range(1, len(data) + 1)
            plt.plot(epochs, data, label=opt_name, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch')
    
    if 'accuracy' in metric.lower():
        plt.ylabel('Accuracy (%)')
    elif 'loss' in metric.lower():
        plt.ylabel('Loss')
    else:
        plt.ylabel(metric)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimizer comparison plot saved: {save_path}")
    
    plt.show()


def plot_multiple_metrics(results, experiment_name=None, save_dir='results/loss_curves'):
    """
    Plot multiple metrics for optimizer comparison
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_name}_" if experiment_name else ""
    
    # Plot test accuracies
    if any('test_accuracies' in result for result in results.values()):
        plot_optimizer_comparison(
            results, 
            metric='test_accuracies',
            title=f"{experiment_name} - Test Accuracy Comparison" if experiment_name else "Test Accuracy Comparison",
            save_path=os.path.join(save_dir, f'{prefix}test_accuracy_comparison_{timestamp}.png')
        )
    
    # Plot training losses
    if any('train_losses' in result for result in results.values()):
        plot_optimizer_comparison(
            results,
            metric='train_losses', 
            title=f"{experiment_name} - Training Loss Comparison" if experiment_name else "Training Loss Comparison",
            save_path=os.path.join(save_dir, f'{prefix}training_loss_comparison_{timestamp}.png')
        )


def create_summary_table(results):
    """
    Create a summary table of optimizer results
    """
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Optimizer':<15} {'Final Acc (%)':<15} {'Best Acc (%)':<15} {'Training Time (s)':<18}")
    print("-"*80)
    
    for opt_name, result in results.items():
        final_acc = result.get('final_accuracy', 0)
        best_acc = max(result.get('test_accuracies', [0]))
        training_time = result.get('training_time', 0)
        
        print(f"{opt_name:<15} {final_acc:<15.2f} {best_acc:<15.2f} {training_time:<18.2f}")
    
    print("="*80)