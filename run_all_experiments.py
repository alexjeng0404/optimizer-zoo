import sys
import os
import importlib
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_all_results_to_json(all_results):
    """保存所有實驗結果到JSON"""
    from utils.save_results import save_results_to_json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for exp_name, results in all_results.items():
        filename = f'results/logs/{exp_name}_results_{timestamp}.json'
        save_results_to_json(results, filename)

def create_comprehensive_summary(all_results):
    """創建綜合總結"""
    from utils.save_results import create_experiment_summary
    create_experiment_summary(all_results)

def run_all_experiments():
    """
    Run all experiments sequentially with error handling
    """
    print("=" * 60)
    print("STARTING ALL OPTIMIZER EXPERIMENTS")
    print("=" * 60)
    
    start_time = time.time()
    success_count = 0
    all_results = {}
    
    experiments = [
        ("toy_function", "🎯 Toy Function"),
        ("mnist_logreg", "📊 Logistic Regression"), 
        ("mnist_mlp", "🧠 MLP")
    ]
    
    for module_name, display_name in experiments:
        print(f"\n{display_name} EXPERIMENT")
        print("-" * 40)
        try:
            # 動態導入模塊
            module = importlib.import_module(f'experiments.{module_name}')
            results = module.main()  # 獲取返回的結果
            
            if results:
                all_results[module_name] = results
                success_count += 1
                print(f"✅ {display_name} completed")
            else:
                print(f"⚠️ {display_name} returned no results")
                
        except Exception as e:
            print(f"❌ {display_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 統一保存所有結果
    if all_results:
        save_all_results_to_json(all_results)
        create_comprehensive_summary(all_results)
        print(f"\n💾 All results saved successfully!")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed: {success_count}/{len(experiments)} experiments")
    print(f"⏰ Total time: {total_time:.2f} seconds")
    
    if success_count == len(experiments):
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check the errors above.")
    
    return all_results

if __name__ == "__main__":
    run_all_experiments()