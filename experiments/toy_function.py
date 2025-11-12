import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

# Import from utils
from utils.plot_utils import plot_optimizer_comparison, create_summary_table
from utils.save_results import save_results_to_json, save_training_plots

# Import all optimizers
from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.adagrad import Adagrad
from optimizers.rmsprop import Rmsprop
from optimizers.adam import Adam
from optimizers.adamw import AdamW
from optimizers.amsgrad import AMSGrad
from optimizers.radam import RAdam
from optimizers.adabelief import AdaBelief
from optimizers.lookahead import Lookahead


def test_optimizers_on_toy_function():
    """
    Test optimizers on simple quadratic function f(x) = x^2 + 3x + 2
    """
    torch.manual_seed(42)
    
    # Test function f(x) = x^2 + 3x + 2
    def f(x):
        return x**2 + 3*x + 2

    # Initialize parameters for each optimizer
    initial_params = {}
    optimizer_names = ['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSProp', 
                      'Adam', 'AdamW', 'AMSGrad', 'RAdam', 'AdaBelief']
    
    for name in optimizer_names:
        initial_params[name] = torch.tensor([5.0], requires_grad=True)

    # Lookahead needs special handling
    lookahead_param = torch.tensor([5.0], requires_grad=True)

    losses = {name: [] for name in optimizer_names}
    losses['Lookahead(Adam)'] = []

    # Create optimizers
    optimizers_dict = {
        'SGD': SGD([initial_params['SGD']], lr=0.1),
        'Momentum': Momentum([initial_params['Momentum']], lr=0.1, beta=0.9),
        'Nesterov': Nesterov([initial_params['Nesterov']], lr=0.1, beta=0.9),
        'Adagrad': Adagrad([initial_params['Adagrad']], lr=0.1),
        'RMSProp': Rmsprop([initial_params['RMSProp']], lr=0.1),
        'Adam': Adam([initial_params['Adam']], lr=0.1),
        'AdamW': AdamW([initial_params['AdamW']], lr=0.1, weight_decay=0.01),
        'AMSGrad': AMSGrad([initial_params['AMSGrad']], lr=0.1),
        'RAdam': RAdam([initial_params['RAdam']], lr=0.1),
        'AdaBelief': AdaBelief([initial_params['AdaBelief']], lr=0.1),
        'Lookahead(Adam)': Lookahead(Adam([lookahead_param], lr=0.1))
    }

    # Training loop
    num_iters = 100
    for name, opt in optimizers_dict.items():
        if name == 'Lookahead(Adam)':
            current_param = lookahead_param
        else:
            current_param = initial_params[name]
        
        # Reset parameter
        current_param.data = torch.tensor([5.0])
        if current_param.grad is not None:
            current_param.grad.zero_()
        
        # Clear previous losses
        if name in losses:
            losses[name] = []
        else:
            losses['Lookahead(Adam)'] = []
        
        for t in range(num_iters):
            opt.zero_grad()
            loss = f(current_param)
            losses[name].append(loss.item())
            loss.backward()
            opt.step()

    return losses, optimizers_dict, initial_params, lookahead_param


def plot_toy_function_results(losses):
    """
    Plot results for toy function optimization
    """
    # Plot 1: Loss curves
    plt.figure(figsize=(12, 8))
    for name, loss_values in losses.items():
        plt.plot(loss_values, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss f(x)')
    plt.title('Optimizer Comparison on f(x) = x² + 3x + 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Convergence speed (log scale)
    plt.figure(figsize=(12, 8))
    optimal_loss = -0.25  # f(-1.5) = 0.25 is the minimum
    
    for name, loss_values in losses.items():
        convergence = [abs(loss - optimal_loss) for loss in loss_values]
        plt.semilogy(convergence, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('|f(x) - f(x*)| (log scale)')
    plt.title('Convergence Speed Comparison (Log Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_final_results(optimizers_dict, initial_params, lookahead_param):
    """
    Print final optimization results
    """
    def f(x):
        return x**2 + 3*x + 2

    print("\n" + "="*60)
    print("FINAL RESULTS - Toy Function Optimization")
    print("="*60)
    
    for name, opt in optimizers_dict.items():
        if name == 'Lookahead(Adam)':
            final_x = lookahead_param.item()
            final_loss = f(lookahead_param).item()
        else:
            final_x = initial_params[name].item()
            final_loss = f(initial_params[name]).item()
        
        optimal_x = -1.5
        distance_to_opt = abs(final_x - optimal_x)
        
        print(f"{name:15s}: x = {final_x:8.4f}, f(x) = {final_loss:8.4f}, |x-x*| = {distance_to_opt:8.4f}")


def main():
    """
    Main function for toy function experiment
    """
    print("Toy Function Optimizer Comparison")
    print("=" * 40)
    
    # Run the experiment
    losses, optimizers_dict, initial_params, lookahead_param = test_optimizers_on_toy_function()
    
    # Print results
    print_final_results(optimizers_dict, initial_params, lookahead_param)
    
    # Plot results
    plot_toy_function_results(losses)
    
    # Create summary for convergence
    results_summary = {}
    optimal_loss = -0.25
    
    for name, loss_values in losses.items():
        final_loss = loss_values[-1]
        convergence_iter = None
        for i, loss in enumerate(loss_values):
            if abs(loss - optimal_loss) < 0.01:  # Within 1% of optimal
                convergence_iter = i
                break
        
        results_summary[name] = {
            'final_loss': final_loss,
            'convergence_iteration': convergence_iter if convergence_iter else len(loss_values),
            'distance_to_optimal': abs(final_loss - optimal_loss)
        }
    
    print("\nConvergence Summary:")
    print(f"{'Optimizer':<15} {'Final Loss':<12} {'Conv. Iter':<12} {'Dist to Opt':<12}")
    print("-" * 60)
    for name, summary in results_summary.items():
        print(f"{name:<15} {summary['final_loss']:<12.4f} "
              f"{summary['convergence_iteration']:<12} {summary['distance_to_optimal']:<12.4f}")
    
    # prepare result
    results_dict = {}
    for name, loss_values in losses.items():
        results_dict[name] = {
            'losses': loss_values,
            'final_loss': loss_values[-1] if loss_values else None
        }
    
    # save JSON
    save_results_to_json(results_dict, 'results/logs/toy_function_results.json')
    
    # save diagram
    save_training_plots(results_dict, 'toy_function')
    
    print("🎯 Toy function experiment completed and results saved!")
    
    return results_dict


if __name__ == "__main__":
    main()