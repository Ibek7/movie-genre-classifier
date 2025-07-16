#!/usr/bin/env python3
"""
Quick test script to verify optimized training pipeline works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.train import train_and_save_models
from sklearn.metrics import accuracy_score, classification_report
import joblib

def test_optimized_pipeline():
    """Test the optimized training pipeline with the integrated improvements"""
    
    print("🚀 Testing Optimized Training Pipeline")
    print("=" * 50)
    
    # Define paths
    data_path = "data/processed/cleaned_plots.csv"
    vec_path = "models/test_vectorizer_optimized.joblib"
    model_paths = {
        "nb": "models/test_nb_optimized.joblib",
        "lr": "models/test_lr_optimized.joblib"
    }
    
    try:
        # Train with optimized parameters
        print("📊 Training models with optimization...")
        X_test, y_test = train_and_save_models(
            data_path=data_path,
            vec_path=vec_path,
            model_paths=model_paths,
            max_features=5000,      # From notebook optimization
            min_genre_samples=100   # From notebook optimization
        )
        
        # Test the saved models
        print("\n🧪 Testing saved models...")
        
        # Load models
        vectorizer = joblib.load(vec_path)
        nb_model = joblib.load(model_paths["nb"])
        lr_model = joblib.load(model_paths["lr"])
        
        print(f"✅ Vectorizer: {vectorizer.max_features} features")
        print(f"✅ NB Classes: {len(nb_model.classes_)}")
        print(f"✅ LR Classes: {len(lr_model.classes_)}")
        
        # Quick predictions
        nb_pred = nb_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        
        nb_acc = accuracy_score(y_test, nb_pred)
        lr_acc = accuracy_score(y_test, lr_pred)
        
        print(f"\n📈 Performance Results:")
        print(f"   Naive Bayes Accuracy: {nb_acc:.3f}")
        print(f"   Logistic Regression Accuracy: {lr_acc:.3f}")
        
        # Test single prediction
        test_plot = "A thrilling action movie with explosions and car chases through the city streets"
        test_vectorized = vectorizer.transform([test_plot])
        
        nb_single = nb_model.predict(test_vectorized)[0]
        lr_single = lr_model.predict(test_vectorized)[0]
        
        print(f"\n🎬 Sample Prediction Test:")
        print(f"   Plot: {test_plot[:50]}...")
        print(f"   NB Prediction: {nb_single}")
        print(f"   LR Prediction: {lr_single}")
        
        print(f"\n🎉 Integration test SUCCESSFUL!")
        print(f"✅ Optimized pipeline is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_pipeline()
    exit(0 if success else 1)
