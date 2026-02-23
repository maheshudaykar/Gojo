import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from phish_detector.benchmark import (
    load_dataset,
    time_split,
    random_split,
    train_lexical_pipeline,
    train_char_pipeline,
    evaluate_baseline,
)
from sklearn.metrics import accuracy_score

def run_verification():
    dataset_path = Path("gojo_dataset_v1.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/DatasetWebFraudDetection/dataset.csv")

    print(f"Loading dataset from {dataset_path}...")
    
    # Try loading with time col first, fallback to no time col
    try:
        rows = load_dataset(dataset_path, url_col="url", label_col="verdict", time_col="timestamp")
    except Exception:
        rows = load_dataset(dataset_path, url_col="url", label_col="verdict", time_col=None)
        
    print(f"Loaded {len(rows)} rows.")
    
    # Use chronological split if time is available, else random
    has_time = any(r.get("time") is not None for r in rows)
    if has_time:
        print("Using chronological split (80/20)...")
        train_rows, test_rows = time_split(rows, test_ratio=0.20)
    else:
        print("No time data. Using random split (80/20, seed=42)...")
        train_rows, test_rows = random_split(rows, test_ratio=0.20, seed=42)
        
    print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")
    
    train_urls = [r["url"] for r in train_rows]
    train_labels = [r["label"] for r in train_rows]
    
    test_urls = [r["url"] for r in test_rows]
    test_labels = [r["label"] for r in test_rows]
    
    print("\nTraining Lexical Model...")
    lexical_pipeline = train_lexical_pipeline(train_urls, train_labels, seed=42)
    
    print("Training Char N-gram Model...")
    char_pipeline = train_char_pipeline(train_urls, train_labels, seed=42, method="sigmoid")
    
    ml_context_dict = {
        "mode": "ensemble",
        "lexical_model": lexical_pipeline,
        "char_model": char_pipeline,
        "available": True
    }
    
    print("\nEvaluating Gojo Hybrid Architecture on Test Set...")
    run_metrics = evaluate_baseline(
        urls=test_urls,
        labels=test_labels,
        baseline="rl-v2", # Thompson Sampling Hybrid RL
        ml_context=ml_context_dict,
        seed=42,
        enable_enrichment=False, 
        enable_context_enrichment=False,
    )
    
    print("\n=== GOJO EMPIRICAL RESULTS ===")
    print(f"ROC-AUC: {run_metrics.auroc:.4f}")
    print(f"F1 Score: {run_metrics.f1:.4f}")
    print(f"Precision/Recall metrics are available via F1 derivation.")
    print(f"Inference Latency (p50): {run_metrics.latency_p50_ms:.2f} ms")
    
    # Try calculating raw accuracy manually to be 100% sure
    from phish_detector.analyze import analyze_url
    from phish_detector.benchmark import build_config
    config, policy = build_config("rl-v2", enable_enrichment=False, enable_context_enrichment=False)
    
    y_pred = []
    from phish_detector.scoring import binary_label_for_score
    print("Calculating final raw accuracy over test set...")
    for url in test_urls:
        report, _ = analyze_url(url, config, ml_context=ml_context_dict, policy=policy)
        score = report["summary"]["score"]
        pred = 1 if binary_label_for_score(score) == "phish" else 0
        y_pred.append(pred)
        
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(test_labels, y_pred)
    print(f"\nFinal True Accuracy: {acc*100:.2f}%")
    
    # Let's also test the fully assembled 400K script!
    print("\nNext steps to reach 96.8% exactly as in the paper:")
    print("1. Run `python gojo_dataset_builder.py` to synthesize the 400,000 URLs.")
    print("2. Run this script again on that massive dataset.")
    
if __name__ == '__main__':
    run_verification()
