import json
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from utils.plot_utils import plot_multiple_metrics

def save_results_to_json(results, filename):
    """
    Save results to JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert any torch tensors to Python values
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_tensors(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {filename}")
    return filename

def save_training_plots(results, experiment_name):
    """
    Save training loss and accuracy plots
    """

    plot_multiple_metrics(results, experiment_name)
    
    print(f"✅ Training plots saved for {experiment_name}")

def create_experiment_summary(all_results):
    """
    Create a comprehensive summary of all experiments
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }
    
    for exp_name, results in all_results.items():
        summary['experiments'][exp_name] = {}
        
        for opt_name, opt_results in results.items():
            if isinstance(opt_results, dict):
                summary['experiments'][exp_name][opt_name] = {
                    'final_accuracy': opt_results.get('final_accuracy', None),
                    'training_time': opt_results.get('training_time', None),
                    'best_accuracy': max(opt_results.get('test_accuracies', [0])) if 'test_accuracies' in opt_results else None,
                    'final_loss': opt_results.get('train_losses', [0])[-1] if 'train_losses' in opt_results else None
                }
    
    # Save summary
    summary_path = 'results/logs/experiment_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Experiment summary saved: {summary_path}")
    return summary