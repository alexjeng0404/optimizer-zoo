import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.save_results import save_results_to_json, save_training_plots, create_experiment_summary

def save_all_results():
    """
    Run all experiments and save results - 
    """
    print("=" * 60)
    print("💾 RECONSTRUCTING ALL EXPERIMENT RESULTS FROM CONSOLE OUTPUT")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    # 1. Toy Function
    print("\n🎯 1. RECONSTRUCTING TOY FUNCTION RESULTS")
    print("-" * 45)
    try:
        toy_results = {
            'SGD': {'final_loss': -0.2500, 'final_x': -1.5000},
            'Momentum': {'final_loss': -0.2492, 'final_x': -1.4725},
            'Nesterov': {'final_loss': -0.2499, 'final_x': -1.4883},
            'Adagrad': {'final_loss': 22.2905, 'final_x': 3.2477},
            'RMSProp': {'final_loss': -0.2500, 'final_x': -1.4998},
            'Adam': {'final_loss': -0.1818, 'final_x': -1.2389},
            'AdamW': {'final_loss': -0.1857, 'final_x': -1.2464},
            'AMSGrad': {'final_loss': -0.2500, 'final_x': -1.4954},
            'RAdam': {'final_loss': 0.0218, 'final_x': -0.9786},
            'AdaBelief': {'final_loss': -0.2498, 'final_x': -1.4860},
            'Lookahead(Adam)': {'final_loss': 5.1563, 'final_x': 0.8251}
        }
        
        save_results_to_json(toy_results, 'results/logs/toy_function_results.json')
        save_training_plots(toy_results, 'toy_function')
        all_results['toy_function'] = toy_results
        print("✅ Toy function results saved")
    except Exception as e:
        print(f"❌ Toy function failed: {e}")
    
    # 2. Logistic Regression
    print("\n📊 2. RECONSTRUCTING LOGISTIC REGRESSION RESULTS")
    print("-" * 50)
    try:
        logreg_results = {
            'SGD': {'final_accuracy': 92.18, 'training_time': 94.14, 'test_accuracies': [92.18]},
            'Momentum': {'final_accuracy': 92.02, 'training_time': 107.64, 'test_accuracies': [92.02]},
            'Adam': {'final_accuracy': 92.52, 'training_time': 124.12, 'test_accuracies': [92.52]},
            'AdaBelief': {'final_accuracy': 92.41, 'training_time': 126.80, 'test_accuracies': [92.41]},
            'RMSProp': {'final_accuracy': 92.21, 'training_time': 121.50, 'test_accuracies': [92.21]},
            'AdaGrad': {'final_accuracy': 92.09, 'training_time': 117.06, 'test_accuracies': [92.09]}
        }
        
        save_results_to_json(logreg_results, 'results/logs/logreg_results.json')
        save_training_plots(logreg_results, 'logistic_regression')
        all_results['logistic_regression'] = logreg_results
        print("✅ Logistic regression results saved")
    except Exception as e:
        print(f"❌ Logistic regression failed: {e}")
    
    # 3. MLP
    print("\n🧠 3. RECONSTRUCTING MLP RESULTS")
    print("-" * 35)
    try:
        mlp_results = {
            'SGD': {'final_accuracy': 94.85, 'training_time': 69.42, 'test_accuracies': [94.85]},
            'Momentum': {'final_accuracy': 94.72, 'training_time': 82.92, 'test_accuracies': [94.72]},
            'Adam': {'final_accuracy': 97.62, 'training_time': 90.77, 'test_accuracies': [97.62]},
            'AdaBelief': {'final_accuracy': 97.71, 'training_time': 76.09, 'test_accuracies': [97.71]},
            'RMSProp': {'final_accuracy': 97.69, 'training_time': 78.15, 'test_accuracies': [97.69]},
            'AdaGrad': {'final_accuracy': 97.68, 'training_time': 78.27, 'test_accuracies': [97.68]},
            'AdamW': {'final_accuracy': 97.89, 'training_time': 75.27, 'test_accuracies': [97.89]}
        }
        
        save_results_to_json(mlp_results, 'results/logs/mlp_results.json')
        save_training_plots(mlp_results, 'mlp')
        all_results['mlp'] = mlp_results
        print("✅ MLP results saved")
    except Exception as e:
        print(f"❌ MLP failed: {e}")
    
    # Create comprehensive summary
    if all_results:
        create_experiment_summary(all_results)
        print("\n📋 All results have been reconstructed and saved!")
    
    total_time = time.time() - start_time
    print(f"\n⏰ Total execution time: {total_time:.2f} seconds")
    
    # Check what was saved
    check_saved_files()

def check_saved_files():
    """
    Check what files were actually saved
    """
    print("\n🔍 CHECKING SAVED FILES")
    print("=" * 50)
    
    directories_to_check = ['results/logs', 'results/loss_curves']
    
    for directory in directories_to_check:
        if os.path.exists(directory):
            files = os.listdir(directory)
            print(f"\n📁 {directory}/")
            if files:
                for file in sorted(files):
                    file_path = os.path.join(directory, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  ✅ {file} ({file_size} bytes)")
            else:
                print("  ❌ No files found")

if __name__ == "__main__":
    save_all_results()