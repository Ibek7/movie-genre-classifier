#!/usr/bin/env python3
"""
Test script for the updated production pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.train import train_and_save_models

def test_production_pipeline():
    """Test the updated training pipeline with optimized parameters"""
    
    print("🚀 Testing Updated Production Pipeline")
    print("=" * 50)
    
    # Define paths
    data_path = "data/processed/cleaned_plots.csv"
    vec_path = "models/test_vectorizer_optimized.joblib"
    model_paths = {
        "nb": "models/test_nb_model.joblib",
        "lr": "models/test_lr_model.joblib"
    }
    
    try:
        # Run the training pipeline with optimized parameters
        X_te, y_test, performance_summary = train_and_save_models(
            data_path=data_path,
            vec_path=vec_path,
            model_paths=model_paths,
            max_features=5000,        # Optimized from notebook
            min_genre_samples=100,    # Optimized from notebook
            test_size=0.2,
            random_state=42
        )
        
        print("\n🎉 SUCCESS: Pipeline completed successfully!")
        print(f"📊 LR Accuracy: {performance_summary['model_performance']['logistic_regression']['accuracy']:.3f}")
        print(f"⚡ LR Training Time: {performance_summary['model_performance']['logistic_regression']['training_time_seconds']:.2f}s")
        print(f"🎯 Production Ready: {performance_summary['production_ready']}")
        
        # Quick prediction test
        import joblib
        print("\n🧪 Testing Quick Prediction...")
        vectorizer = joblib.load(vec_path)
        lr_model = joblib.load(model_paths["lr"])
        
        test_plot = "A thrilling action movie with explosions and car chases through the city."
        test_vectorized = vectorizer.transform([test_plot])
        prediction = lr_model.predict(test_vectorized)[0]
        
        print(f"Test prediction: '{prediction}' for action movie plot")
        print("✅ Pipeline is ready for production!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_pipeline()
    sys.exit(0 if success else 1)
